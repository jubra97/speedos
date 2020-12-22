from src.agents import NStepSurvivalAgent, MultiVoronoiMultiMiniMax, VoronoiMultiMiniMaxAgent, MultiprocessedDepthMultiMiniMax
from src.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(20, 20, 2, agent_classes=[MultiVoronoiMultiMiniMax, NStepSurvivalAgent], verbose=True)
    model.run_model()
