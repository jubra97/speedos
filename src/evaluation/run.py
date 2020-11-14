from src.evaluation.utils import Evaluator
from src.model.agents import OneStepSurvivalAgent, RandomAgent


model_params = {
    "width": 10,
    "height": 10,
    "nb_agents": 2,
    "agent_classes": [OneStepSurvivalAgent, RandomAgent],
}
evaluator = Evaluator(model_params)
evaluator.evaluate(100, save=True)
