import unittest
from datetime import datetime

from mesa import Model

from src.core.agents import NStepSurvivalAgent, RandomAgent, DummyAgent
from src.core.model import SpeedModel
from src.core.utils import *


class TestOutOfBounds(unittest.TestCase):

    def test_in_bounds(self):
        self.assertFalse(out_of_bounds((10, 10), (0, 0)))
        self.assertFalse(out_of_bounds((10, 10), (9, 9)))
        self.assertFalse(out_of_bounds((10, 10), (3, 7)))

    def test_out_of_bounds(self):
        self.assertTrue(out_of_bounds((10, 10), (-1, 0)))
        self.assertTrue(out_of_bounds((10, 10), (0, -1)))
        self.assertTrue(out_of_bounds((10, 10), (0, 10)))
        self.assertTrue(out_of_bounds((10, 10), (10, 0)))
        self.assertTrue(out_of_bounds((10, 10), (10, -1)))


class TestAgentToJson(unittest.TestCase):

    def setUp(self):
        self.model = Model()

    def test_default_params(self):
        agent = DummyAgent(self.model, (5, 23), Direction.UP)
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
        agent = DummyAgent(self.model, (5, 23), Direction.UP, 9, False)
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


class TestModelToJson(unittest.TestCase):

    def setUp(self):
        self.model = SpeedModel(30, 30, 2, [RandomAgent, RandomAgent],
                                initial_agents_params=[{"pos": (10, 10), "direction": Direction.DOWN},
                                                       {"pos": (20, 20), "direction": Direction.LEFT}])

        self.json = {
            "width": 30,
            "height": 30,
            "cells": self.model.cells.copy(),
            "players": {
                "1": {
                    "x": 10,
                    "y": 10,
                    "direction": str(Direction.DOWN),
                    "speed": 1,
                    "active": True
                },
                "2": {
                    "x": 20,
                    "y": 20,
                    "direction": str(Direction.LEFT),
                    "speed": 1,
                    "active": True
                }
            },
            "running": True
        }

    def test_default(self):
        result = model_to_json(self.model)
        self.assertTrue((result["cells"] == self.json["cells"]).all())

        result.pop("cells")
        default_json = self.json.copy()
        default_json.pop("cells")
        self.assertDictEqual(result, default_json)

    def test_trace_aware(self):
        result = model_to_json(self.model, trace_aware=True)
        self.assertTrue((result["cells"] == self.json["cells"]).all())

        result.pop("cells")
        trace_json = self.json.copy()
        trace_json.pop("cells")
        trace_json["players"]["1"]["trace"] = []
        trace_json["players"]["2"]["trace"] = []
        self.assertDictEqual(result, trace_json)

    def test_step(self):
        result = model_to_json(self.model, step=True)
        self.assertTrue((result["cells"] == self.json["cells"]).all())

        result.pop("cells")
        step_json = self.json.copy()
        step_json.pop("cells")
        step_json["step"] = 0
        self.assertDictEqual(result, step_json)


class TestGetState(unittest.TestCase):

    def setUp(self):
        self.model = SpeedModel(30, 30, 2, [RandomAgent, RandomAgent],
                                initial_agents_params=[{"pos": (10, 10), "direction": Direction.DOWN},
                                                       {"pos": (20, 20), "direction": Direction.LEFT}])
        self.own_agent = self.model.active_speed_agents[0]

        self.json = {
            "width": 30,
            "height": 30,
            "cells": self.model.cells.copy(),
            "players": {
                "1": {
                    "x": 10,
                    "y": 10,
                    "direction": str(Direction.DOWN),
                    "speed": 1,
                    "active": True
                },
                "2": {
                    "x": 20,
                    "y": 20,
                    "direction": str(Direction.LEFT),
                    "speed": 1,
                    "active": True
                }
            },
            "running": True,
            "you": 1
        }

    def test_default(self):
        result = get_state(self.model, self.own_agent)
        self.assertTrue((result["cells"] == self.json["cells"]).all())

        result.pop("cells")
        default_json = self.json.copy()
        default_json.pop("cells")
        self.assertDictEqual(result, default_json)

    def test_deadline(self):
        deadline = "2020-12-24T12:59:00Z"
        result = get_state(self.model, self.own_agent, deadline=datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ"))
        self.assertTrue((result["cells"] == self.json["cells"]).all())

        result.pop("cells")
        default_json = self.json.copy()
        default_json.pop("cells")
        default_json["deadline"] = deadline
        self.assertDictEqual(result, default_json)


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


class TestStateToModel(unittest.TestCase):

    def setUp(self):
        self.model = SpeedModel(30, 30, 2, [RandomAgent, RandomAgent],
                                initial_agents_params=[{"pos": (10, 10), "direction": Direction.DOWN},
                                                       {"pos": (20, 20), "direction": Direction.LEFT}])

    def test_default(self):
        result_model = state_to_model(model_to_json(self.model))
        self.assertEqual(result_model.width, self.model.width)
        self.assertEqual(result_model.height, self.model.height)
        self.assertTrue((result_model.cells == self.model.cells).all())
        self.assertEqual(result_model.nb_agents, self.model.nb_agents)
        self.assertEqual(result_model.running, self.model.running)


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
