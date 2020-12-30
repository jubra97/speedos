from src.agents import NStepSurvivalAgent, MultiprocessedVoronoiMultiMiniMaxAgent, VoronoiMultiMiniMaxAgent, \
    BaseMultiMiniMaxAgent
from src.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(60, 60, 2,
                       agent_classes=[MultiprocessedVoronoiMultiMiniMaxAgent, MultiprocessedVoronoiMultiMiniMaxAgent],
                       verbose=True)
    model.run_model()
