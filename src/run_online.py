import datetime
import os
import json

import websockets
import asyncio

from src.model.agents import RandomAgent, MultiMiniMaxAgent
from src.utils import state_to_model

API_KEY = "IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO"


class RunOnline:
    def __init__(self, agent=MultiMiniMaxAgent, save=True):
        self.connection = None
        self.history = []
        self.agent = agent(None, None, None)
        self.save = save
        self.loop = asyncio.get_event_loop()
        self.tasks = []

    async def run(self):
        #self.loop.run_until_complete(self.connect())
        #self.tasks = [asyncio.ensure_future(self.play_game())]
        #self.loop.run_until_complete(asyncio.wait(self.tasks))
        await self.connect()
        return await self.play_game()

    async def connect(self):
        url = "wss://msoll.de/spe_ed?key=" + API_KEY
        self.connection = await websockets.client.connect(url)
        if self.connection.open:
            print(f"Connection established at {datetime.datetime.now()}", flush=True)

    async def play_game(self):
        while True:
            try:
                message = await self.connection.recv()
                message = json.loads(message)
                self.history.append(message)
                #print(message)
                if message["running"] is False:
                    return message
                action = self.agent.act(message)
                if message["players"][str(message["you"])]["active"]:
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    #print(respond)
                    try:
                        await self.connection.send(respond)
                    except Exception as e:
                        print(e)
            except websockets.exceptions.ConnectionClosed:
                print("Connection with server closed.")
                if self.save:
                    original_games_path = os.path.abspath("..") + "\\res\\recordedGames\\"
                    with open(original_games_path + datetime.datetime.now().strftime("%d-%m-%y__%H-%M") + ".json", "w") \
                            as f:
                        json.dump(self.history, f, indent=4)
                break


if __name__ == "__main__":
    print("starting", flush=True)
    runner = RunOnline()
    games = 0
    wins = 0
    while True:
        game = asyncio.get_event_loop().run_until_complete(runner.run())
        games += 1
        if game["running"] and game["players"][str(game["you"])]["active"]:
            wins += 1
        print("current stats: " + str(wins/games), flush=True)
