from gym import Env, spaces
import numpy as np

from src.model.model import SpeedModel
from src.model.agents import AgentDummy
from src.utils import Action, Direction
from src.utils import get_state
from src.rl.gym.rewards import LongSurvivalReward


class SingleAgentSpeedEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width=3, height=3, reward=LongSurvivalReward()):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(len(Action))
        lows = np.asarray([*np.full(width * height, -1), 0, 1])
        highs = np.asarray([*np.full(width * height, 1), len(Direction) - 1, 10])
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.int)
        self.reward = reward
        self.reward_range = reward.reward_range
        self.state = None
        self.model = None
        self.agent = None

        self.reset()

    def seed(self, seed=None):
        self.model.reset_randomizer(seed)
        return [seed]

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=1, #initial_agents_params=[agent_params],
                                agent_classes=[AgentDummy], verbose=False)
        self.seed()
        self.agent = self.model.speed_agents[0]
        self.state = get_state(self.model, self.agent)

        return self.prepared_state

    def step(self, action):
        self.agent.action = Action(action)
        self.model.step()

        new_state = get_state(self.model, self.agent)
        reward = self.reward.payout(self.state, action, new_state)
        done = not self.agent.active
        info = {}
        self.state = new_state

        return self.prepared_state, reward, done, info

    def render(self, mode='ansi'):
        if mode == 'ansi':
            print(self.agent.action, "\n")
            print(self.state["cells"], "\n")
            if not self.agent.active:
                self.model.print_standings()
        else:
            return NotImplementedError

    def close(self):
        pass

    @property
    def prepared_state(self):
        # Todo: Round % 6 counter missing
        # Format: [cells, direction, speed]
        return np.asarray([
            *np.asarray(self.state["cells"]).flatten(),
            Direction[self.state["players"]["1"]["direction"].upper()].value,
            self.state["players"]["1"]["speed"]
        ])
