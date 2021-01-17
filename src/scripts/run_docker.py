import asyncio
import datetime
import json
import os
import websockets

from src.core.agents import LiveAgent


class RunDocker:
    def __init__(self, agent=LiveAgent):
        self.connection = None
        self.agent = agent(None, None, None, server_time_url=os.environ["TIME_URL"])

    async def run(self):
        await self.connect()
        await self.play_game()

    async def connect(self):
        url = os.environ["URL"] + "?key=" + os.environ["KEY"]
        self.connection = await websockets.client.connect(url)
        if self.connection.open:
            print(f"Connection established at {datetime.datetime.now()}")

    async def play_game(self):
        while True:
            try:
                message = await self.connection.recv()
                message = json.loads(message)
                if message["running"] is False or message["players"][str(message["you"])]["active"] is False:
                    break
                else:
                    action = self.agent.act(message)
                    respond = str(action)
                    respond = f'{{"action": "{respond}"}}'
                    try:
                        await self.connection.send(respond)
                    except Exception as e:
                        print(e)
            except websockets.exceptions.ConnectionClosed:
                print("Connection with server closed.")
                break


if __name__ == "__main__":
    runner = RunDocker(agent=LiveAgent)
    asyncio.get_event_loop().run_until_complete(runner.run())
