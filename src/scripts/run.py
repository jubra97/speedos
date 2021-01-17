
from src.core.agents import VoronoiAgent, SlidingWindowVoronoiAgent
from src.core.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(6, 6, 2, agent_classes=[VoronoiAgent, SlidingWindowVoronoiAgent])
    model.run_model()
