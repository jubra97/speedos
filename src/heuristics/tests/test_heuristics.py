import unittest
import numpy as np

from src.heuristics.heuristics import multi_minimax
from src.agents import MultiMiniMaxAgent
from src.model import SpeedModel
from src.utils import Direction, get_state, Action


class TestMultiMiniMax(unittest.TestCase):

    def test_no_gamble(self):
        # two agents facing each other should not gamble and go for the cell between them (in this case)
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
                           agent_classes=[MultiMiniMaxAgent for _ in range(2)])

        # without voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=False)
        action_agent_2 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[1]),
                                       super_pruning=False, use_voronoi=False)
        self.assertEqual(Action.TURN_RIGHT, action_agent_1)
        self.assertEqual(Action.TURN_LEFT, action_agent_2)

        # with voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=True)
        action_agent_2 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[1]),
                                       super_pruning=False, use_voronoi=True)
        self.assertEqual(Action.TURN_RIGHT, action_agent_1)
        self.assertEqual(Action.TURN_LEFT, action_agent_2)

    def test_winning_move(self):
        # There is only one winning move (TURN_RIGHT) for agent 1 in a late endgame. All other actions lead to a loss or
        # draw if agent 2 plays perfectly.
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
                           agent_classes=[MultiMiniMaxAgent for _ in range(2)])

        # without voronoi
        action_agent_1 = multi_minimax(depth=8, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=False)
        self.assertEqual(Action.TURN_RIGHT, action_agent_1)

        # with voronoi
        action_agent_1 = multi_minimax(depth=8, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=True)
        self.assertEqual(Action.TURN_RIGHT, action_agent_1)

    def test_cut_off(self):
        # Agent 1 can cut off agent 2 in one move (only with action SPEED_UP) and win the endgame
        # (voronoi should detect that)
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
        model = SpeedModel(width=7, height=8, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                           agent_classes=[MultiMiniMaxAgent for _ in range(2)])

        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=True)
        self.assertEqual(Action.SPEED_UP, action_agent_1)

    def test_kamikaze(self):
        # Eliminating another agent is better than just dying when death is inevitable
        # Agent 1 can not avoid elimination but can eliminate agent 2 as well with CHANGE_NOTHING
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
                           agent_classes=[MultiMiniMaxAgent for _ in range(2)])

        # without voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=False)
        self.assertEqual(Action.CHANGE_NOTHING, action_agent_1)

        # with voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=True)
        self.assertEqual(Action.CHANGE_NOTHING, action_agent_1)

    @unittest.skip  # not yet implemented
    def test_force_draw(self):
        # Force a draw with kamikaze in a obviously loosing endgame if a save action is chosen
        # Agent 1 can not avoid elimination but can eliminate agent 2 as well with SPEED_UP but could survive one step
        # longer with CHANGE_NOTHING.
        # Assuming that agent 2 is not just eliminating himself, it is best to force the draw.
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
                           agent_classes=[MultiMiniMaxAgent for _ in range(2)])

        # without voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=False)
        self.assertEqual(Action.SPEED_UP, action_agent_1)

        # with voronoi
        action_agent_1 = multi_minimax(depth=2, game_state=get_state(model, model.active_speed_agents[0]),
                                       super_pruning=False, use_voronoi=True)
        self.assertEqual(Action.SPEED_UP, action_agent_1)