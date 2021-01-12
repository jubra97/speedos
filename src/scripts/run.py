from src.core.agents import ParallelVoronoiAgent
from src.core.model import SpeedModel

if __name__ == "__main__":
    model = SpeedModel(6, 6, 2, agent_classes=[ParallelVoronoiAgent, ParallelVoronoiAgent])
    model.run_model()
