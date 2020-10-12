from src.model.model import SpeedModel
import json
import os
from src.model.utils import Direction, state_to_model
from src.model.agents import ValidationAgent

#model = SpeedModel(10, 10, 2)
#model.run_model()


def validate_with_original_game(path_to_game):
    with open(path_to_game, "r") as file:
        game = json.load(file)

    # remove double entries
    compare_agent = -1
    for i in range(1, len(game[-1]["players"])+1):
        if game[-1]["players"][str(i)]["active"]:
            compare_agent = str(i)
            break
    rounds_to_remove = []
    for i in range(len(game)-2):
        if game[i]["players"][compare_agent]["x"] == game[i + 1]["players"][compare_agent]["x"] and \
                game[i]["players"][compare_agent]["y"] == game[i + 1]["players"][compare_agent]["y"]:
            rounds_to_remove.append(i)
    print(f"Removed Index: {rounds_to_remove}")
    for r in reversed(rounds_to_remove):
        del game[r]

    initial_state = game[0]
    print(len(game))
    additional_params = [{"game": game} for i in range(len(initial_state["players"]))]
    model = state_to_model(initial_state, ValidationAgent, additional_params)
    model.run_model()


validate_with_original_game(os.path.abspath("..") + "/res/originalGames/spe_ed-1601974843093.json")