import os
import json

import websockets
import asyncio

from src.model.agents import RandomAgent, MultiMiniMaxAgent
from src.utils import state_to_model
from time import sleep
from datetime import datetime
import statistics
from pathlib import Path


API_KEY = "IXT57ZEJMO6VFKF3KBZFB4LSEXBMWJ72VEYO2B6WT25UOXEIEAEN25XO"


class RunOnline:
    def __init__(self, agent=MultiMiniMaxAgent, save_files_dir=None):
        self.connection = None
        self.history = []
        self.agent = agent(None, None, None)
        self.loop = asyncio.get_event_loop()
        self.tasks = []
        self.save_files_dir = save_files_dir

    async def run(self, run_number):
        await self.connect()
        return await self.play_game(run_number)

    async def connect(self):
        url = "wss://msoll.de/spe_ed?key=" + API_KEY
        self.connection = await websockets.client.connect(url)
        if self.connection.open:
            print(f"Connection established at {datetime.now()}", flush=True)

    async def play_game(self, run_number):
        round = 0
        response_times = []
        place = -1
        while True:
            try:
                message = await self.connection.recv()
                time_msg_recv = datetime.utcnow()
                round += 1
                message = json.loads(message)
                if "deadline" in message:
                    deadline = datetime.strptime(message["deadline"], "%Y-%m-%dT%H:%M:%SZ")
                    r_time = (deadline - time_msg_recv).total_seconds()
                    response_times.append(r_time)
                self.history.append(message)
                if message["running"] is False:
                    if self.save_files_dir is not None:
                        with open(f"{self.save_files_dir}/{run_number}.json", "w") as f:
                            json.dump(self.history, f, indent=4)
                    return message, round, statistics.mean(response_times), place

                if message["players"][str(message["you"])]["active"]:
                    action = self.agent.act(message)
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    if datetime.utcnow() > deadline:
                        print("Missed deadline.")
                    try:
                        await self.connection.send(respond)
                    except Exception as e:
                        print(e)
                else:
                    if place == -1:
                        place = len(list(filter(lambda x: x[1]["active"] is True, message["players"].items())))
            except websockets.exceptions.ConnectionClosed:
                print("Connection with server closed.", flush=True)
                break


def write_result(results_file_path, game_number, game, end_round, avg_r_time, place):
    with open(results_file_path, "a+") as f:
        win = False
        if game["players"][str(game["you"])]["active"]:
            win = True

        line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            game_number,
            win,
            len(game["players"]),
            avg_r_time,
            place,
            "{} x {}".format(game["width"], game["height"]),
            end_round)

        print(line, flush=True)
        f.write(line)
        f.flush()


if __name__ == "__main__":
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = f"./run_{timestamp}/data"
    print("starting", flush=True)
    json_dir = run_dir + "/json_files"
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    runner = RunOnline(agent=MultiMiniMaxAgent, save_files_dir=json_dir)
    games = 0
    wins = 0
    results_file_path = run_dir + "/results.txt"
    while True:
        try:
            games += 1
            game, end_round, avg_r_time, place = asyncio.get_event_loop().run_until_complete(runner.run(games))
            if game["players"][str(game["you"])]["active"]:
                wins += 1
            write_result(results_file_path, games, game, end_round, avg_r_time, place)
            print("current stats: " + str(wins/games), flush=True)
        except Exception as e:
            print(str(e), flush=True)
            sleep(60)
