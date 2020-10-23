from src.model.model import SpeedModel
import json
import os
from src.model.utils import Action, state_to_model
from src.model.agents import ValidationAgent, RandomAgent, OneStepSurvivalAgent
#import websockets
import asyncio
import random


# API KEY: IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO
#model = SpeedModel(10, 10, 4)
# with open((os.path.abspath("..") + "/res/originalGames/test_game.json"), "r") as file:
#     game = json.load(file)
# model = state_to_model(game, False, agent_classes=[RandomAgent, RandomAgent, RandomAgent])
#model.run_model()

def run_against_original_game():
    async def runner():
        first = True
        running = True
        async with websockets.connect(
                "wss://msoll.de/spe_ed?key=IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO") as websocket:
            while running:
                message = await websocket.recv()
                state = json.loads(message)
                running = state["running"]
                print(running)
                if first:
                    first = False
                print(message)
                await websocket.send(f'{{"action": "{random.choice(list(Action))}"}}')

    loop = asyncio.get_event_loop()
    loop.create_task(runner())
    loop.run_forever()


#validate_with_original_game(os.path.abspath("..") + "/res/originalGames/test_game.json")

# run_against_original_game()

