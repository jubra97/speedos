from src.agents import NStepSurvivalAgent, MultiVoronoiMultiMiniMax, VoronoiMultiMiniMaxAgent, \
    MultiprocessedDepthMultiMiniMax, BaseMultiMiniMaxAgent
from src.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(60, 60, 2, agent_classes=[MultiVoronoiMultiMiniMax, VoronoiMultiMiniMaxAgent], verbose=True)
    model.run_model()
