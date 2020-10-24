import unittest
from mesa import Model
from src.utils import *
from src.model.agents import SpeedAgent


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
