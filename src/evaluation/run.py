from src.agents import NStepSurvivalAgent
from src.evaluation.utils import Evaluator

model_params = {
    "width": 20,
    "height": 20,
    "nb_agents": 3,
    "agent_classes": [NStepSurvivalAgent, NStepSurvivalAgent, NStepSurvivalAgent],
}
evaluator = Evaluator(model_params)
evaluator.fair_start_evaluate(3, save=True)
