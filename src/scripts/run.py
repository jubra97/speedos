from src.agents import NStepSurvivalAgent
from src.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(10, 10, 2, agent_classes=[NStepSurvivalAgent for i in range(2)])
    model.run_model()
