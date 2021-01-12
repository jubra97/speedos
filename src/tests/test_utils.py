import unittest

import numpy as np
from mesa import Model

from src.agents import NStepSurvivalAgent
from src.model import SpeedAgent
from src.model import SpeedModel
from src.utils import *
from src.voronoi import voronoi, voronoi_for_reduced_opponents


class TestAgentToJson(unittest.TestCase):

    def setUp(self):
        self.model = Model()

    def test_default_params(self):
        agent = SpeedAgent(self.model, (5, 23), Direction.UP)
        self.assertEqual(
            agent_to_json(agent),
            {
                "x": 5,
                "y": 23,
                "direction": "up",
                "speed": 1,
                "active": True
            }
        )

    def test_non_default_params(self):
        agent = SpeedAgent(self.model, (5, 23), Direction.UP, 9, False)
        self.assertEqual(
            agent_to_json(agent),
            {
                "x": 5,
                "y": 23,
                "direction": "up",
                "speed": 9,
                "active": False
            }
        )


class TestArgMaxes(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(arg_maxes([], indices=None), [])

    def test_one_max(self):
        self.assertEqual(arg_maxes([1], indices=None), [0])
        self.assertEqual(arg_maxes([-1, 0, 5, 4], indices=None), [2])
        self.assertEqual(arg_maxes([-1, -4, -3], indices=None), [0])
        self.assertEqual(arg_maxes([-1, -4, -3], indices=['a', 'b', 'c']), ['a'])

    def test_multiple_maxes(self):
        self.assertEqual(arg_maxes([-1, 5, 5, 4], indices=None), [1, 2])
        self.assertEqual(arg_maxes([-1, -1, -10, -1], indices=None), [0, 1, 3])
        self.assertEqual(arg_maxes([-1, -1, -10, -1], indices=['a', 'b', 'c', 'd']), ['a', 'b', 'd'])


class TestReduceStateToSlidingWindow(unittest.TestCase):

    def test_size(self):
        initial_agents_params = [
            {"pos": (0, 0), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (1, 1), "direction": Direction.LEFT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=10, height=10, nb_agents=2, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])

        state = model_to_json(model)
        state["you"] = '1'
        new_state = reduce_state_to_sliding_window(state, 1, 3, 0)

        self.assertEqual(new_state['width'], 4)
        self.assertEqual(new_state['height'], 4)

    def test_size_2(self):
        initial_agents_params = [
            {"pos": (0, 0), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (9, 9), "direction": Direction.LEFT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=10, height=10, nb_agents=2, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])

        state = model_to_json(model)
        state["you"] = '1'
        new_state = reduce_state_to_sliding_window(state, 9, 3, 3)

        self.assertEqual(new_state['width'], 10)
        self.assertEqual(new_state['height'], 10)

    def test_agent_removed(self):
        initial_agents_params = [
            {"pos": (0, 0), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (1, 1), "direction": Direction.LEFT, "deterministic": True, "depth": 1},
            {"pos": (9, 9), "direction": Direction.LEFT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=10, height=10, nb_agents=2, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])

        state = model_to_json(model)
        state["you"] = '1'
        new_state = reduce_state_to_sliding_window(state, 1, 3, 3)

        self.assertEqual(len(new_state['players']), 2)


class TestModelToJson(unittest.TestCase):
    def test_model_to_json(self):
        model = SpeedModel(width=10, height=10, nb_agents=2, agent_classes=[NStepSurvivalAgent for _ in range(2)])

        state = model_to_json(model, True, True)
        self.assertEqual(len(state['players']), 2)
        self.assertEqual(state['width'], 10)
        self.assertEqual(state['height'], 10)
        self.assertTrue(state['running'])

        for _ in range(5):
            model.step()

        state = model_to_json(model, True, True)
        self.assertEqual(len(state['players']), 2)
        self.assertEqual(state['width'], 10)
        self.assertEqual(state['height'], 10)
        self.assertTrue(state['running'])
        self.assertEqual(state['step'], 5)

        state = model_to_json(model, False, False)
        self.assertFalse('step' in state.keys())












