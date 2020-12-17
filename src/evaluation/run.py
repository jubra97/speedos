from src.agents import NStepSurvivalAgent, RandomAgent
from src.evaluation.utils import Evaluator

model_params = {
    "width": 10,
    "height": 10,
    "nb_agents": 2,
    "agent_classes": [NStepSurvivalAgent, RandomAgent],
}
evaluator = Evaluator(model_params)
evaluator.evaluate(100, save=True)
