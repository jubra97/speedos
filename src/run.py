from src.model.model import SpeedModel
import json
import os
from src.model.utils import Direction
from src.model.agents import ValidationAgent

#model = SpeedModel(10, 10, 2)
#model.run_model()


def create_model_on_original_game(path):
    with open(path, "r") as file:
        game = json.load(file)
    width = game[0]["width"]
    height = game[0]["height"]
    nb_agents = len(game[0]["players"])
    initial_params = []
    for values in game[0]["players"].values():
        initial_params.append({
            "pos": (values["x"], values["y"]),
            "direction":  Direction[values["direction"].upper()],
            "game": game  # game[0] is just for initialization
            })
    model = SpeedModel(width, height, nb_agents, initial_params, [ValidationAgent for i in range(nb_agents)])
    model.run_model()

create_model_on_original_game(os.path.abspath("..") + "/res/originalGames/spe_ed-1602439201755.json")