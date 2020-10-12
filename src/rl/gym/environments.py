import numpy as np

from gym import Env, spaces

from src.model.model import SpeedModel
from src.model.agents import SpeedAgent
from src.model.utils import Action, Direction
from src.model.utils import get_state
from src.rl.gym.rewards import LongSurvivalReward


class SingleAgentSpeedEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width=3, height=3, reward=LongSurvivalReward()):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(len(Action))
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(width, height), dtype=int)
        self.observation_space = spaces.Box(low=0, high=2, shape=(2,), dtype=int)
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
        # Todo: This is simplified for testing atm (The initial position and direction should not be predefined)
        agent_params = {
            "pos": (2, 2),
            "direction": Direction.UP
        }
        self.model = SpeedModel(self.width, self.height, nb_agents=1, initial_agents_params=[agent_params],
                                agent_classes=[RLAgent], verbose=False)
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
            print(self.state["cells"], "\n")
            print(self.agent.action, "\n")
            if not self.agent.active:
                self.model.print_standings()
        else:
            return NotImplementedError

    def close(self):
        pass

    @property
    def prepared_state(self):
        #return self.state["cells"]
        return self.agent.pos


class RLAgent(SpeedAgent):
    """
    Agent dummy for reinforcement learning.
    It doesn't choose and set an action since the Gym environment controls the execution.
    """

    def act(self, state):
        return None

    def step(self):
        pass
