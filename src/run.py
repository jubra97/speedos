from src.model.model import SpeedModel
from src.model.agents import OneStepSurvivalAgent

model = SpeedModel(10, 10, 2, agent_classes=[OneStepSurvivalAgent for i in range(2)])
model.run_model()
