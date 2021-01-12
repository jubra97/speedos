from multiprocessing.spawn import freeze_support

from src.core.agents import MultiMinimaxAgent, VoronoiAgent, SlidingWindowVoronoiAgent, ClosestOpponentsVoronoiAgent
from src.evaluation.utils import Evaluator

if __name__ == '__main__':
    freeze_support()
    for i in range(100):
        model_params = {
            "width": 60,
            "height": 60,
            "nb_agents": 5,
            "agent_classes": [MultiMinimaxAgent, VoronoiAgent, SlidingWindowVoronoiAgent, SlidingWindowVoronoiAgent,
                              ClosestOpponentsVoronoiAgent],
            "initial_agents_params": [{}, {}, {"min_sliding_window_size": 20, "sliding_window_size_offset": 5},
                                      {"min_sliding_window_size": 15, "sliding_window_size_offset": 3}, {}]
        }
        evaluator = Evaluator(model_params)
        evaluator.fair_start_evaluate(5, save=True, random_move_time=True)
