
import timeit

if __name__ == "__main__":
    setup1 = """
from src.core.model import SpeedModel
from src.core.agents import RandomAgent
from src.core.utils import Direction
from src.core.voronoi import voronoi_for_reduced_opponents
initial_agents_params = [{"pos": (40, 40), "direction": Direction.UP}, {"pos": (10, 10), "direction": Direction.DOWN}]
model = SpeedModel(50, 50, 2,
                   agent_classes=[RandomAgent, RandomAgent], initial_agents_params=initial_agents_params)
    """

    setup2 = """
from src.core.model import SpeedModel
from src.core.agents import RandomAgent
from src.core.utils import Direction
from src.core.voronoi_cython_unchanged import voronoi_for_reduced_opponents
initial_agents_params = [{"pos": (40, 40), "direction": Direction.UP}, {"pos": (10, 10), "direction": Direction.DOWN}]
model = SpeedModel(50, 50, 2,
                   agent_classes=[RandomAgent, RandomAgent], initial_agents_params=initial_agents_params)
    
    """

    setup3 = """
from src.core.model import SpeedModel
from src.core.agents import RandomAgent
from src.core.utils import Direction
from src.core.voronoi_cython import voronoi_for_reduced_opponents
initial_agents_params = [{"pos": (40, 40), "direction": Direction.UP}, {"pos": (10, 10), "direction": Direction.DOWN}]
model = SpeedModel(50, 50, 2,
                   agent_classes=[RandomAgent, RandomAgent], initial_agents_params=initial_agents_params)
        """

    t = timeit.Timer("voronoi_for_reduced_opponents(model, 1, 2, False)", setup=setup1)
    print(t.repeat(5, 100))
    t = timeit.Timer("voronoi_for_reduced_opponents(model, 1, 2, False)", setup=setup2)
    print(t.repeat(5, 100))
    t = timeit.Timer("voronoi_for_reduced_opponents(model, 1, 2, False)", setup=setup3)
    print(t.repeat(5, 100))