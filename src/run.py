from src.model.model import SpeedModel
import json
import os
from src.model.utils import Action, state_to_model
from src.model.agents import ValidationAgent, RandomAgent, OneStepSurvivalAgent
#import websockets
import asyncio
import random


# API KEY: IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO
model = SpeedModel(10, 10, 4)
# with open((os.path.abspath("..") + "/res/originalGames/test_game.json"), "r") as file:
#     game = json.load(file)
# model = state_to_model(game, False, agent_classes=[RandomAgent, RandomAgent, RandomAgent])
model.run_model()




#validate_with_original_game(os.path.abspath("..") + "/res/originalGames/test_game.json")

# run_against_original_game()

