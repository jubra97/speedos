import asyncio
import datetime
import json
import os

import websockets

from src.agents import LiveAgent

API_KEY = "IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO"


class RunOnline:
    def __init__(self, agent=LiveAgent, save=False):
        self.connection = None
        self.history = []
        self.agent = agent(None, None, None)
        self.save = save

    async def run(self):
        await self.connect()
        await self.play_game()

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
                if message["players"][str(message["you"])]["active"]:
                    action = self.agent.act(message)
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    try:
                        await self.connection.send(respond)
                    except Exception as e:
                        print(e)
            except websockets.exceptions.ConnectionClosed:
                print("Connection with server closed.")
                if self.save:
                    original_games_path = os.path.abspath("../..") + "\\res\\recordedGames\\"
                    with open(original_games_path + datetime.datetime.now().strftime("%d-%m-%y__%H-%M") + ".json", "w") \
                            as f:
                        json.dump(self.history, f, indent=4)
                break


if __name__ == "__main__":
    runner = RunOnline(agent=LiveAgent)
    asyncio.get_event_loop().run_until_complete(runner.run())
