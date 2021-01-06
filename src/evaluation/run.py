from src.agents import NStepSurvivalAgent, RandomAgent, MultiprocessedVoronoiMultiMiniMaxAgent, \
    MultiprocessedSlidingWindowVoronoiMultiMiniMaxAgent
from src.evaluation.utils import Evaluator

if __name__ == "__main__":
    model_params = {
        "width": 20,
        "height": 20,
        "nb_agents": 3,
        "agent_classes": [MultiprocessedSlidingWindowVoronoiMultiMiniMaxAgent, MultiprocessedVoronoiMultiMiniMaxAgent,
                          NStepSurvivalAgent],
    }
    evaluator = Evaluator(model_params)
    evaluator.fair_start_evaluate(6, save=True)
