import datetime
import os
import json

import websockets
import asyncio

from src.model.agents import RandomAgent, MultiMiniMaxAgent
from src.utils import state_to_model
from time import sleep
from datetime import datetime
import statistics


API_KEY = "IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO"


class RunOnline:
    def __init__(self, agent=MultiMiniMaxAgent, save=False):
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
        round = 0
        response_times = []
        while True:
            try:
                message = await self.connection.recv()
                round += 1
                message = json.loads(message)
                deadline = datetime.strptime(message["deadline"], '%b %d %Y %I:%M%p')
                r_time = (deadline - datetime.utcnow()).total_seconds()
                response_times.append(r_time)
                self.history.append(message)
                #print(message)
                if message["running"] is False:
                    return message, round, statistics.mean(response_times)
                if message["players"][str(message["you"])]["active"]:
                    action = self.agent.act(message)
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    #print(respond, flush=True)
                    try:
                        await self.connection.send(respond)
                    except Exception as e:
                        print(e)
            except websockets.exceptions.ConnectionClosed:
                print("Connection with server closed.", flush=True)
                if self.save:
                    original_games_path = os.path.abspath("..") + "\\res\\recordedGames\\"
                    with open(original_games_path + datetime.datetime.now().strftime("%d-%m-%y__%H-%M") + ".json", "w") \
                            as f:
                        json.dump(self.history, f, indent=4)
                break


def write_result(results_file_path, game_number, game, end_round, avg_r_time):
    with open(results_file_path, "a+") as f:
        win = False
        if game["players"][str(game["you"])]["active"]:
            win = True

        f.write("{}\t{}\t{}\t{}\n".format(
            game_number,
            win,
            avg_r_time,
            "{} x {}".format(game["width"], game["height"]),
            end_round))
        f.flush()


if __name__ == "__main__":
    print("starting", flush=True)
    runner = RunOnline()
    games = 0
    wins = 0
    results_file_path = "results.txt"
    while True:
        try:
            game, end_round, avg_r_time = asyncio.get_event_loop().run_until_complete(runner.run())
            games += 1
            if game["players"][str(game["you"])]["active"]:
                wins += 1
            write_result(results_file_path, games, game, end_round, avg_r_time)
            print("current stats: " + str(wins/games), flush=True)
        except Exception as e:
            print(e)
            sleep(60)
