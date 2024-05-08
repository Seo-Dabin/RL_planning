from typing import Dict, List

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from shapely.geometry.base import CAP_STYLE

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.geometry.transform import rotate_angle
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.ml.ml_agent import mlAgent
from nuplan.planning.simulation.observation.ml.ml_states import mlLeadAgentState
from nuplan.planning.simulation.observation.ml.utils import path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap

# added 24 04 29
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, ProgressStateSE2
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath

UniquemlAgents = Dict[str, mlAgent]

import numpy as np
import math
from typing import List

from GameFormerPlanner.GameFormer.predictor import GameFormer
import torch
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
# from GameFormerPlanner.Planner.planner import Planner
from GameFormerPlanner.Planner.observation import general_observation_adapter, observation_adapter

def generate_Progress_path(x_values : np.array, y_values : np.array) -> List[ProgressStateSE2]:
    assert len(x_values) == len(y_values), "length of x & y points error"
    num_points = len(x_values)
    headings = np.arctan2(np.cos(x_values), np.ones_like(x_values)) # Derivative of sin is cos
    arc_length = np.cumsum(np.sqrt(np.diff(x_values, prepend=x_values[0])**2 + np.diff(y_values, prepend=y_values[0])**2))

    ProgressPath = [
        ProgressStateSE2(progress=arc_length[i], x=x_values[i], y=y_values[i], heading=headings[i])
        for i in range(num_points)
    ]

    return ProgressPath

# Example usage:

class MLAgentManager:
    """ml smart-agents manager."""

    def __init__(self, agents: UniquemlAgents, agent_occupancy: OccupancyMap, map_api: AbstractMap):
        """
        Constructor for mlAgentManager.
        :param agents: A dictionary pairing the agent's token to it's ml representation.
        :param agent_occupancy: An occupancy map describing the spatial relationship between agents.
        :param map_api: AbstractMap API
        """
        self.agents: UniquemlAgents = agents
        self.agent_occupancy = agent_occupancy
        self._map_api = map_api

    def propagate_agents(
        self,
        ego_state: EgoState,
        tspan: float,
        iteration: int,
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
        open_loop_detections: List[TrackedObject],
        radius: float,
        history: SimulationHistoryBuffer
    ) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param open_loop_detections: A list of open loop detections the ml agents should be responsive to.
        :param radius: [m] The radius around the ego state
        """
        self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
        track_ids = []
        for track in open_loop_detections:
            track_ids.append(track.track_token)
            self.agent_occupancy.insert(track.track_token, track.box.geometry)

        self._filter_agents_out_of_range(ego_state, radius)

        ########################## load game-former model : Completed ##########################
        model = GameFormer(encoder_layers=3, decoder_levels=2)
        model_path = '../../../../../../GameFormerPlanner/training_log/Exp2_100epoch/model_epoch_100_valADE_2.3318.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():

                agent.plan_route(traffic_light_status)
                # Add stop lines into occupancy map if they are impacting the agent
                stop_lines = self._get_relevant_stop_lines(agent, traffic_light_status)
                # Keep track of the stop lines that were inserted. This is to remove them for each agent
                inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)

                ###################### get observation for each agent : TBD ######################
                route_roadblock_ids = self._map_api._vector_map['lane_groups_polygons']['fid']
                center_agent_n = 1
                ### Need to be developed
                # input = general_observation_adapter(history, traffic_light_status, self._map_api, route_roadblock_ids, center_agent_n, device='cpu')
                decoder_outputs, agent_plan = model(input)
                if agent_plan.is_cuda:
                    agent_plan = agent_plan.cpu()
                agent_plan = agent_plan.detach().numpy()
                ### Need to be developed
                # x_values = agent_plan.x
                # y_values = agent_plan.y

                ###################### rewrite agent trajectory : Completed ######################
                x_values = [ego_state.center.x + 10, ego_state.center.x + 11, ego_state.center.x + 12] # Toy example
                y_values = [ego_state.center.y + 10, ego_state.center.y + 11, ego_state.center.y + 12] # Toy example

                agent._path = InterpolatedPath(generate_Progress_path(np.array(x_values),np.array(y_values)))
                agent._requires_state_update = True
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
                self.agent_occupancy.remove(inactive_stop_line_tokens)
        self.agent_occupancy.remove(track_ids)

        ################################ some information ################################
        # input = {'ego_agent_past' : ego_agent_past,
        #          'neighbor_agents_past' : neighbor_agents_past,
        #          'map_lanes' : map_lanes,
        #          'map_crosswalks' : map_crosswalks,
        #          'route_lanes' : route_lanes}

        # ego_agent_past : nn.LSTM("7", 256, 2, batch_first=True)
        # neighbor_agents_past : nn.LSTM("11", 256, 2, batch_first=True)
        # neighbors = inputs['neighbor_agents_past']
        # / encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        # map_lanes : nn.Sequential(nn.Linear("7", 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256)) / PositionalEncoding(max_len="50")
        # map_crosswalks : nn.Sequential(nn.Linear("3", 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256)) / PositionalEncoding(max_len="30")
        # route_lanes : nn.Sequential(nn.Linear("3", 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256)) / PositionalEncoding(max_len="50")

        # self._future_len = 80

        # agent_path_x = agent_trajectories[0, agent_n, 0, 0] # ?, ?, modality, time_steps, [x,y,h,?]
        # agent_path_y = agent_trajectories[0, agent_n, 0, 1]

        # decoder_outputs: dict_keys(
        #     ['level_0_interactions', 'level_0_scores', 'level_1_interactions', 'level_1_scores', 'level_2_interactions',
        #      'level_2_scores'])
        # ego_plan: torch.Size([1, 80, 3])
        # history._ego_state_buffer
        # history._observations_buffer

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> DetectionsTracks:
        """
        Returns all agents as DetectionsTracks.
        :param iteration: the current simulation iteration.
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as DetectionsTracks.
        """
        return DetectionsTracks(
            TrackedObjects(
                [
                    agent.get_agent_with_planned_trajectory(num_samples, sampling_time)
                    for agent in self.agents.values()
                    if agent.is_active(iteration)
                ]
            )
        )

    def _filter_agents_out_of_range(self, ego_state: EgoState, radius: float = 100) -> None:
        """
        Filter out agents that are out of range.
        :param ego_state: The ego state used as the center of the given radius
        :param radius: [m] The radius around the ego state
        """
        if len(self.agents) == 0:
            return

        agents: npt.NDArray[np.int32] = np.array([agent.to_se2().point.array for agent in self.agents.values()])
        distances = cdist(np.expand_dims(ego_state.center.point.array, axis=0), agents)
        remove_indices = np.argwhere(distances.flatten() > radius)
        remove_tokens = np.array(list(self.agents.keys()))[remove_indices.flatten()]

        # Remove agents which are out of scope
        self.agent_occupancy.remove(remove_tokens)
        for token in remove_tokens:
            self.agents.pop(token)

    def _get_relevant_stop_lines(
        self, agent: mlAgent, traffic_light_status: Dict[TrafficLightStatusType, List[str]]
    ) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.
        :param agent: The ml agent of interest.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        relevant_lane_connectors = list(
            {segment.id for segment in agent.get_route()} & set(traffic_light_status[TrafficLightStatusType.RED])
        )
        lane_connectors = [
            self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in relevant_lane_connectors
        ]
        return [stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines]

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f"stop_line_{stop_line.id}"
            if not self.agent_occupancy.contains(stop_line_token):
                self.agent_occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)

        return stop_line_tokens
