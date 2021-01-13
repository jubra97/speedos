from collections import defaultdict

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid

from src.core.agents import ParallelVoronoiAgent, VoronoiAgent
from src.core.model import SpeedAgent, AgentTrace, AgentTraceCollision
from src.core.model import SpeedModel
from src.core.utils import Direction


class CustomCanvasGrid(CanvasGrid):
    """
    CanvasGrid with top left origin.
    """

    def render(self, model):
        grid_state = defaultdict(list)
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                cell_objects = model.grid.get_cell_list_contents([(x, y)])
                for obj in cell_objects:
                    portrayal = self.portrayal_method(obj)
                    if portrayal:
                        portrayal["x"] = x
                        portrayal["y"] = model.grid.height - 1 - y
                        grid_state[portrayal["Layer"]].append(portrayal)

        return grid_state


def agent_portrayal(agent):
    portrayal = {
        "Filled": "true",
        "Layer": 0
    }
    if isinstance(agent, SpeedAgent):
        portrayal = dict(portrayal, **{
            "Shape": "circle",
            "Color": COLOR_PALETTE[agent.unique_id - 1],
            "r": 0.95
        })
    elif type(agent) is AgentTrace:
        portrayal = dict(portrayal, **{
            "Shape": "rect",
            "Color": COLOR_PALETTE[agent.origin.unique_id - 1],
            "w": 0.95,
            "h": 0.95
        })
    if type(agent) is AgentTraceCollision:
        portrayal = dict(portrayal, **{
            "Shape": "circle",
            "Color": " brown",
            "r": 0.95
        })
    return portrayal


if __name__ == "__main__":
    # Parameters
    WIDTH = 40
    HEIGHT = 40
    COLOR_PALETTE = [
        'green',
        'blue',
        'red',
        'black',
        'pink',
        'orange'
    ]
    initial_agents_params = [{"pos": (29, 29), "direction": Direction.UP},
                             {"pos": (10, 10), "direction": Direction.DOWN},
                             {"pos": (29, 10), "direction": Direction.DOWN},
                             {"pos": (10, 29), "direction": Direction.UP}]
    model_params = {
        "width": WIDTH,
        "height": HEIGHT,
        "agent_classes": [ParallelVoronoiAgent, ParallelVoronoiAgent, VoronoiAgent, VoronoiAgent],
        "nb_agents": UserSettableParameter('slider', 'Amount of Agents', value=4, min_value=1, max_value=6, step=1),
        "save": True,
        "initial_agents_params": initial_agents_params
    }
    grid = CustomCanvasGrid(agent_portrayal, WIDTH, HEIGHT, 700, 700)

    # create and launch server instance
    server = ModularServer(SpeedModel,
                           [grid],
                           "Speed",
                           model_params)
    server.port = 8521
    server.launch()
