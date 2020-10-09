from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules.TextVisualization import TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer
from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy
from src.model.model import SpeedModel
from src.model.agents import SpeedAgent, AgentTrace


# Needed to resolve a NotImplementedError
set_event_loop_policy(WindowsSelectorEventLoopPolicy())


def agent_portrayal(agent):
    portrayal = None
    if isinstance(agent, SpeedAgent):
        portrayal = {"Shape": "circle",
                     "Color": COLOR_PALETTE[agent.unique_id - 1],
                     "Filled": "true",
                     "Layer": 0,
                     "r": 0.95}
    elif type(agent) is AgentTrace:
        portrayal = {"Shape": "rect",
                     "Color": COLOR_PALETTE[agent.origin.unique_id - 1],
                     "Filled": "true",
                     "Layer": 0,
                     "w": 0.95,
                     "h": 0.95}
    return portrayal


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
model_params = {
    "width": WIDTH,
    "height": HEIGHT,
    "nb_agents": UserSettableParameter('slider', 'Amount of Agents', value=4, min_value=1, max_value=6, step=1)
}
grid = CanvasGrid(agent_portrayal, WIDTH, HEIGHT, 700, 700)

# create and launch server instance
server = ModularServer(SpeedModel,
                       [grid],
                       "Speed",
                       model_params)
server.port = 8521
server.launch()
