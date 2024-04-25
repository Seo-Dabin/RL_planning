from collections import defaultdict
from typing import Dict, List, Optional, Type

from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.ml.ml_agent_manager import MLAgentManager
from nuplan.planning.simulation.observation.ml.ml_agents_builder import build_ml_agents_on_map_rails
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class MLAgents(AbstractObservation):
    """
    Simulate agents based on ml policy.
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        open_loop_detections_types: List[str],
        scenario: AbstractScenario,
        minimum_path_length: float = 20,
        planned_trajectory_samples: Optional[int] = None,
        planned_trajectory_sample_interval: Optional[float] = None,
        radius: float = 100,
    ):
        """
        Constructor for mlAgents

        :param target_velocity: [m/s] Desired velocity in free traffic
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front
        :param accel_max: [m/s^2] maximum acceleration
        :param decel_max: [m/s^2] maximum deceleration (positive value)
        :param scenario: scenario
        :param open_loop_detections_types: The open-loop detection types to include.
        :param minimum_path_length: [m] The minimum path length to maintain.
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param radius: [m] Only agents within this radius around the ego will be simulated.
        """
        self.current_iteration = 0

        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max
        self._scenario = scenario
        self._open_loop_detections_types: List[TrackedObjectType] = []
        self._minimum_path_length = minimum_path_length
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._radius = radius

        # Prepare ml agent manager
        self._ml_agent_manager: Optional[MLAgentManager] = None
        self._initialize_open_loop_detection_types(open_loop_detections_types)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._ml_agent_manager = None

    def _initialize_open_loop_detection_types(self, open_loop_detections: List[str]) -> None:
        """
        Initializes open-loop detections with the enum types from TrackedObjectType
        :param open_loop_detections: A list of open-loop detections types as strings
        :return: A list of open-loop detections types as strings as the corresponding TrackedObjectType
        """
        for _type in open_loop_detections:
            try:
                self._open_loop_detections_types.append(TrackedObjectType[_type])
            except KeyError:
                raise ValueError(f"The given detection type {_type} does not exist or is not supported!")

    def _get_ml_agent_manager(self) -> MLAgentManager:
        """
        Create ml agent manager in case it does not already exists
        :return: MLAgentManager
        """
        if not self._ml_agent_manager:
            agents, agent_occupancy = build_ml_agents_on_map_rails(
                self._target_velocity,
                self._min_gap_to_lead_agent,
                self._headway_time,
                self._accel_max,
                self._decel_max,
                self._minimum_path_length,
                self._scenario,
                self._open_loop_detections_types,
            )
            self._ml_agent_manager = MLAgentManager(agents, agent_occupancy, self._scenario.map_api)

        return self._ml_agent_manager

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        detections = self._get_ml_agent_manager().get_active_agents(
            self.current_iteration, self._planned_trajectory_samples, self._planned_trajectory_sample_interval
        )
        if self._open_loop_detections_types:
            open_loop_detections = self._get_open_loop_track_objects(self.current_iteration)
            detections.tracked_objects.tracked_objects.extend(open_loop_detections)
        return detections

    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        tspan = next_iteration.time_s - iteration.time_s
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(self.current_iteration)

        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)

        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))

        ego_state, _ = history.current_state

        if not self._ml_agent_manager:
            agents, agent_occupancy = build_ml_agents_on_map_rails(
                self._target_velocity,
                self._min_gap_to_lead_agent,
                self._headway_time,
                self._accel_max,
                self._decel_max,
                self._minimum_path_length,
                self._scenario,
                self._open_loop_detections_types,
            )
            self._ml_agent_manager = MLAgentManager(agents, agent_occupancy, self._scenario.map_api)

        self._ml_agent_manager.propagate_agents(
            ego_state,
            tspan,
            self.current_iteration,
            traffic_light_status,
            self._get_open_loop_track_objects(self.current_iteration),
            self._radius,
        )

        # self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
        # track_ids = []
        # for track in open_loop_detections:
        #     track_ids.append(track.track_token)
        #     self.agent_occupancy.insert(track.track_token, track.box.geometry)
        #
        # self._filter_agents_out_of_range(ego_state, radius)
        #
        # for agent_token, agent in self.agents.items():
        #     if agent.is_active(iteration) and agent.has_valid_path():
        #         ###################################################################################
        #         ### 중략 -relative_distance, projected_velocity, length_rear 다양한 case 문으로 계산 ###
        #         ###################################################################################
        #         agent.propagate(
        #             mlLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear),
        #             tspan,
        #         )
        #         self.agent_occupancy.set(agent_token, agent.projected_footprint)
        #         self.agent_occupancy.remove(inactive_stop_line_tokens)
        # self.agent_occupancy.remove(track_ids)

    def _get_open_loop_track_objects(self, iteration: int) -> List[TrackedObject]:
        """
        Get open-loop tracked objects from scenario.
        :param iteration: The simulation iteration.
        :return: A list of TrackedObjects.
        """
        detections = self._scenario.get_tracked_objects_at_iteration(iteration)
        return detections.tracked_objects.get_tracked_objects_of_types(self._open_loop_detections_types)  # type: ignore
