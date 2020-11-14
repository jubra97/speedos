import unittest
from src.rl.q_agent import *
from gym import Env, spaces


class TestQTableEncoding(unittest.TestCase):

    def test_scalar_low_high(self):
        q_table = QTable(spaces.Box(low=-5, high=5, shape=(2,), dtype=np.int), spaces.Discrete(1))
        self.assertEqual(q_table.encode_observation([-5, -5]), 0)
        self.assertEqual(q_table.encode_observation([-3, -2]), 35)
        self.assertEqual(q_table.encode_observation([5, 5]), 120)

    def test_non_scalar_low_high(self):
        q_table = QTable(spaces.Box(low=np.asarray((-5, 0)), high=np.asarray((-1, 1)), shape=(2,), dtype=np.int),
                         spaces.Discrete(1))
        self.assertEqual(q_table.encode_observation([-5, 0]), 0)
        self.assertEqual(q_table.encode_observation([-3, 1]), 7)
        self.assertEqual(q_table.encode_observation([-1, 1]), 9)

    def test_multidimensional_shape(self):
        q_table = QTable(spaces.Box(low=1, high=5, shape=(2, 2), dtype=np.int), spaces.Discrete(1))
        self.assertEqual(q_table.encode_observation([[1, 1], [1, 1]]), 0)
        self.assertEqual(q_table.encode_observation([[3, 1], [2, 1]]), 27)
        self.assertEqual(q_table.encode_observation([[5, 5], [5, 5]]), 624)
