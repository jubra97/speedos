import datetime
import os
import json

import websockets
import asyncio

from src.model.agents import RandomAgent

API_KEY = "IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO"


class RunOnline:
    def __init__(self, agent=RandomAgent, save=True):
        self.connection = None
        self.history = []
        self.agent = agent(None, None, None)
        self.save = save
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.connect())
        self.tasks = [asyncio.ensure_future(self.play_game())]
        loop.run_until_complete(asyncio.wait(self.tasks))

    async def connect(self):
        url = "wss://msoll.de/spe_ed?key=" + API_KEY
        self.connection = await websockets.client.connect(url)
        if self.connection.open:
            print(f"Connection established at {datetime.datetime.now()}")

    async def play_game(self):
        while True:
            try:
                message = await self.connection.recv()
                message = json.loads(message)
                self.history.append(message)
                print(message)
                action = self.agent.act(message)
                if message["players"][str(message["you"])]["active"]:
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    print(respond)
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
    RunOnline()