import unittest

import numpy as np

from src.core.agents import NStepSurvivalAgent
from src.core.model import SpeedModel
from src.core.utils import Direction
from src.core.voronoi import voronoi, voronoi_for_reduced_opponents, Particle, surrounding_cells


class TestVoronoi(unittest.TestCase):

    def test_2_agents(self):
        initial_agents_params = [
            {"pos": (2, 4), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (6, 1), "direction": Direction.LEFT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=10, height=10, nb_agents=2, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, model.active_speed_agents[0].unique_id)
        self.assertEqual({0: 2, 1: 48, 2: 50}, region_sizes)
        self.assertEqual(False, is_endgame)

        # run for 5 steps and tests again
        for _ in range(5):
            model.step()
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, model.active_speed_agents[0].unique_id)
        self.assertEqual({0: 12, 1: 53, 2: 35}, region_sizes)
        self.assertEqual(False, is_endgame)

    def test_6_agents(self):
        initial_agents_params = [
            {"pos": (0, 0), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (23, 3), "direction": Direction.LEFT, "deterministic": True, "depth": 1},
            {"pos": (5, 35), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (26, 40), "direction": Direction.UP, "deterministic": True, "depth": 1},
            {"pos": (17, 8), "direction": Direction.RIGHT, "deterministic": True, "depth": 1},
            {"pos": (39, 39), "direction": Direction.UP, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=50, height=50, nb_agents=6, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(6)])
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, model.active_speed_agents[0].unique_id)
        self.assertEqual({-1: 83, 0: 6, 1: 126, 2: 477, 3: 457, 4: 422, 5: 349, 6: 580}, region_sizes)
        self.assertEqual(False, is_endgame)

        # run for 5 steps and tests again
        for _ in range(5):
            model.step()
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, model.active_speed_agents[0].unique_id)
        self.assertEqual({-1: 76, 0: 32, 2: 484, 3: 456, 4: 437, 5: 493, 6: 522}, region_sizes)
        self.assertEqual(False, is_endgame)

    def test_endgame(self):
        # creating a model where agent 1 is in an endgame
        cells = np.array([
            [1, 0, 0, 2, 0],
            [1, 0, 0, 2, 0],
            [1, 1, 0, 2, 2]
        ])
        initial_agents_params = [
            {"pos": (1, 2), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (4, 2), "direction": Direction.RIGHT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=5, height=3, nb_agents=2, cells=cells, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, 1)
        self.assertEqual({0: 8, 1: 5, 2: 2}, region_sizes)
        self.assertEqual(True, is_endgame)

        # run one steps and tests again
        model.step()
        particle_cells, region_sizes, is_endgame, _ = voronoi(model, 1)
        self.assertEqual({0: 10, 1: 4, 2: 1}, region_sizes)
        self.assertEqual(True, is_endgame)

    def test_voronoi_for_reduced_opponents(self):
        initial_agents_params = [
            {"pos": (2, 4), "direction": Direction.DOWN, "deterministic": True, "depth": 1},
            {"pos": (6, 1), "direction": Direction.LEFT, "deterministic": True, "depth": 1}
        ]
        model = SpeedModel(width=10, height=10, nb_agents=2, initial_agents_params=initial_agents_params,
                           agent_classes=[NStepSurvivalAgent for _ in range(2)])

        particle_cells, region_sizes, is_endgame = \
            voronoi_for_reduced_opponents(model, model.active_speed_agents[0].unique_id,
                                          model.active_speed_agents[1].unique_id,
                                          False)
        self.assertEqual({1: 48, 2: 50}, region_sizes)
        self.assertEqual(False, is_endgame)

        # run for 5 steps and tests again
        for _ in range(5):
            model.step()

        particle_cells, region_sizes, is_endgame = \
            voronoi_for_reduced_opponents(model, model.active_speed_agents[0].unique_id,
                                          model.active_speed_agents[1].unique_id,
                                          False)
        self.assertEqual({1: 53, 2: 35}, region_sizes)
        self.assertEqual(False, is_endgame)


class TestSurroundingCells(unittest.TestCase):

    def test_surrounding_cells(self):
        parent = Particle((5, 5), 1, Direction.DOWN)
        particles = surrounding_cells(parent, 20, 20)
        all_pos = [p.position for p in particles]

        self.assertEqual(len(particles), 3)
        self.assertTrue((6, 5) in all_pos)
        self.assertTrue((4, 5) in all_pos)
        self.assertTrue((5, 6) in all_pos)
        self.assertFalse((5, 4) in all_pos)

        parent = Particle((0, 0), 1, Direction.DOWN)
        particles = surrounding_cells(parent, 20, 20)
        all_pos = [p.position for p in particles]

        self.assertEqual(len(particles), 2)
        self.assertTrue((0, 1) in all_pos)
        self.assertTrue((1, 0) in all_pos)
        self.assertFalse((1, 1) in all_pos)
