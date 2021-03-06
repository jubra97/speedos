import unittest

import numpy as np

from src.core.agents import MultiMinimaxAgent, VoronoiAgent, ClosestOpponentsVoronoiAgent, NStepSurvivalAgent, \
    SlidingWindowVoronoiAgent, ParallelSlidingWindowVoronoiAgent, ParallelVoronoiAgent
from src.core.model import SpeedModel
from src.core.utils import Direction, get_state, Action


class TestAgents(unittest.TestCase):

    def setUp(self):
        self.agent_classes = [MultiMinimaxAgent, VoronoiAgent, ParallelVoronoiAgent, ClosestOpponentsVoronoiAgent,
                              SlidingWindowVoronoiAgent, ParallelSlidingWindowVoronoiAgent]

    def test_no_gamble(self):
        # two agents facing each other should not gamble and go for the cell between them (in this case)
        for agent_cls in self.agent_classes:
            cells = np.array([
                [0, 1, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ])
            initial_agents_params = [
                {"pos": (1, 0), "direction": Direction.RIGHT},
                {"pos": (3, 0), "direction": Direction.LEFT}
            ]
            model = SpeedModel(width=5, height=5, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                               agent_classes=[agent_cls, agent_cls])

            action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=2,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 0]))
            action_agent_2 = model.active_speed_agents[0].multi_minimax(depth=2,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 1]))
            self.assertEqual(Action.TURN_RIGHT, action_agent_1)
            self.assertEqual(Action.TURN_LEFT, action_agent_2)

    def test_winning_move(self):
        # There is only one winning move (TURN_RIGHT) for agent 1 in a late endgame. All other actions lead to a loss or
        # draw if agent 2 plays perfectly.
        for agent_cls in self.agent_classes:
            cells = np.array([
                [0, 0, 0, 0, 0, 0, 2],
                [0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1]
            ])
            initial_agents_params = [
                {"pos": (1, 1), "direction": Direction.LEFT},
                {"pos": (6, 0), "direction": Direction.LEFT}
            ]
            model = SpeedModel(width=7, height=4, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                               agent_classes=[agent_cls for _ in range(2)])

            action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=8,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 0]))
            self.assertEqual(Action.TURN_RIGHT, action_agent_1)

    def test_cut_off(self):
        # Agent 1 can cut off agent 2 in one move (only with action SPEED_UP) and win the endgame
        # (voronoi should detect that)
        for agent_cls in self.agent_classes:
            if agent_cls != MultiMinimaxAgent:
                cells = np.array([
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [2, 2, 2, 0, 2, 2, 2],
                    [2, 2, 2, 0, 0, 0, 2],
                    [2, 2, 2, 2, 0, 0, 2],
                    [2, 2, 2, 2, 2, 2, 2],
                ])
                initial_agents_params = [
                    {"pos": (5, 3), "direction": Direction.LEFT},
                    {"pos": (3, 6), "direction": Direction.UP}
                ]
                model = SpeedModel(width=7, height=8, nb_agents=2, cells=cells,
                                   initial_agents_params=initial_agents_params,
                                   agent_classes=[agent_cls for _ in range(2)])

                game_state = get_state(model, model.active_speed_agents[0])
                action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=2, game_state=game_state)
                self.assertEqual(Action.SPEED_UP, action_agent_1)

    def test_kamikaze(self):
        # Eliminating another agent is better than just dying when death is inevitable
        # Agent 1 can not avoid elimination but can eliminate agent 2 as well with CHANGE_NOTHING
        for agent_cls in self.agent_classes:
            cells = np.array([
                [0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2],
                [0, 2, 2, 2, 2],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ])
            initial_agents_params = [
                {"pos": (0, 3), "direction": Direction.UP},
                {"pos": (1, 2), "direction": Direction.LEFT}
            ]
            model = SpeedModel(width=5, height=5, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                               agent_classes=[agent_cls for _ in range(2)])

            action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=2,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 0]))
            self.assertEqual(Action.CHANGE_NOTHING, action_agent_1)

    @unittest.skip  # not yet implemented
    def test_force_draw(self):
        # Force a draw with kamikaze in a obviously loosing endgame if a save action is chosen
        # Agent 1 can not avoid elimination but can eliminate agent 2 as well with SPEED_UP but could survive one step
        # longer with CHANGE_NOTHING.
        # Assuming that agent 2 is not just eliminating himself, it is best to force the draw.
        for agent_cls in self.agent_classes:
            cells = np.array([
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2],
                [0, 2, 2, 2, 2],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ])
            initial_agents_params = [
                {"pos": (0, 4), "direction": Direction.UP},
                {"pos": (1, 2), "direction": Direction.LEFT}
            ]
            model = SpeedModel(width=5, height=6, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                               agent_classes=[agent_cls for _ in range(2)])

            action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=2,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 0]))
            self.assertEqual(Action.SPEED_UP, action_agent_1)

    def test_jumping_out(self):
        # Agent should pick speed up to jump over wall
        for agent_cls in self.agent_classes:
            cells = np.array([
                [0, 0, 2, 2, 2],
                [0, 0, 2, 0, 0],
                [2, 1, 2, 0, 0],
                [2, 0, 2, 0, 0],
                [2, 0, 2, 0, 0],
                [2, 0, 2, 0, 0],
                [2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ])
            initial_agents_params = [
                {"pos": (1, 2), "direction": Direction.DOWN, "speed": 1},
                {"pos": (4, 0), "direction": Direction.LEFT}
            ]
            model = SpeedModel(width=5, height=12, nb_agents=2, cells=cells,
                               initial_agents_params=initial_agents_params,
                               agent_classes=[agent_cls, NStepSurvivalAgent])

            model.active_speed_agents[0].game_step = 4
            model.schedule.steps = 4

            action_agent_1 = model.active_speed_agents[0].multi_minimax(depth=6,
                                                                        game_state=get_state(model,
                                                                                             model.active_speed_agents[
                                                                                                 0]))

            self.assertEqual(Action.SPEED_UP, action_agent_1)
