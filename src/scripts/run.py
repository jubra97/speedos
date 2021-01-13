from src.core.agents import ParallelVoronoiAgent, MultiMinimaxAgent, VoronoiAgent
from src.core.model import SpeedModel
from src.core.utils import Direction

if __name__ == "__main__":
    initial_agents_params = [{"pos": (40, 40), "direction": Direction.UP},
                             {"pos": (10, 10), "direction": Direction.DOWN}]

    model = SpeedModel(50, 50, 2,
                       agent_classes=[ParallelVoronoiAgent, ParallelVoronoiAgent], initial_agents_params=initial_agents_params)
    model.run_model()