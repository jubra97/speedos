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
with open((os.path.abspath("..") + "/res/originalGames/test_game.json"), "r") as file:
    game = json.load(file)
model = state_to_model(game, False, agent_classes=[RandomAgent, RandomAgent, RandomAgent])
model.run_model()


def validate_with_original_game(path_to_game):
    with open(path_to_game, "r") as file:
        game = json.load(file)

    # remove double entries
    compare_agent = -1
    for i in range(1, len(game[-1]["players"]) + 1):
        if game[-1]["players"][str(i)]["active"]:
            compare_agent = str(i)
            break
    rounds_to_remove = []
    for i in range(len(game) - 2):
        if game[i]["players"][compare_agent]["x"] == game[i + 1]["players"][compare_agent]["x"] and \
                game[i]["players"][compare_agent]["y"] == game[i + 1]["players"][compare_agent]["y"]:
            rounds_to_remove.append(i)
    # print(f"Removed Index: {rounds_to_remove}")
    for r in reversed(rounds_to_remove):
        del game[r]

    initial_state = game[0]
    # print(len(game))
    additional_params = [{"game": game, "checker_agent": int(compare_agent)} for i in range(len(initial_state["players"]))]
    model = state_to_model(initial_state, False, [ValidationAgent for i in range(len(initial_state["players"]))], additional_params)
    model.run_model()


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


#validate_with_original_game(os.path.abspath("..") + "/res/originalGames/spe_ed-1601974843093.json")
#run_against_original_game()

