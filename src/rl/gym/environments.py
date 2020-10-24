from gym import Env, spaces
import numpy as np

from src.model.model import SpeedModel
from src.model.agents import AgentDummy
from src.utils import Action, Direction
from src.utils import get_state
from src.rl.gym.rewards import LongSurvivalReward


class SingleAgentSpeedEnv(Env):
    """
    A Gym environment of Speed with only the learning agent and no opponents.
    An optimal agent learns to survive as long as possible (fill as many cells as possible before being eliminated)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width=10, height=10, reward=LongSurvivalReward(), state_preparation_fn=None):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(len(Action))
        lows = np.asarray([*np.full(width * height, -1), 0, 1])
        highs = np.asarray([*np.full(width * height, 1), len(Direction) - 1, 10])
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.int)
        self.reward = reward
        self.reward_range = reward.reward_range
        self.state_preparation_fn = state_preparation_fn
        self.state = None
        self.model = None
        self.agent = None

        self.reset()

    def seed(self, seed=None):
        self.model.reset_randomizer(seed)
        return [seed]

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=1, agent_classes=[AgentDummy], verbose=False)
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
        if self.state_preparation_fn:
            return self.state_preparation_fn(self.state)
        else:
            # Todo: Round % 6 counter missing
            # Format: [cells, direction, speed]
            agent_id = self.agent.unique_id
            cells = self.state["cells"]
            cells[self.state["players"][str(agent_id)]["y"]][self.state["players"][str(agent_id)]["x"]] = 2
            return np.asarray([
                *np.asarray(cells).flatten(),
                Direction[self.state["players"][str(agent_id)]["direction"].upper()].value,
                self.state["players"][str(agent_id)]["speed"]
            ])


class HeuristicsSpeedEnv(SingleAgentSpeedEnv):
    """
    A Gym environment of Speed with a given set of static heuristic opponents.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, agent_classes, width=10, height=10, reward=LongSurvivalReward(), state_preparation_fn=None):
        self.agent_classes = agent_classes
        super(HeuristicsSpeedEnv, self).__init__(width, height, reward, state_preparation_fn)

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=len(self.agent_classes),
                                agent_classes=self.agent_classes, verbose=False)
        self.seed()
        self.agent = self.model.speed_agents[0]
        self.state = get_state(self.model, self.agent)

        return self.prepared_state


class SelfPlaySpeedEnv(SingleAgentSpeedEnv):
    """
    A Gym environment of Speed that lets the agent play against itself.
    The step()-method expects an array of actions (one for each agent instance) and returns lists of new states,
    rewards, dones and infos (with new_state[i], reward[i], done[i] and info[i]) being the transition for agent i.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, nb_agents=5, width=10, height=10, reward=LongSurvivalReward(), state_preparation_fn=None):
        self.nb_agents = nb_agents
        super(SelfPlaySpeedEnv, self).__init__(width, height, reward, state_preparation_fn)

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=self.nb_agents,
                                agent_classes=[AgentDummy for _ in range(self.nb_agents)], verbose=False)
        self.seed()
        self.agent = self.model.speed_agents
        self.state = list(map(get_state, [self.model for _ in range(self.nb_agents)], self.agent))

        return self.prepared_state

    def step(self, actions):
        for i in range(actions):
            self.agent[i].action = Action(actions[i])
        self.model.step()

        new_state = list()
        reward = list()
        done = list()
        info = list()
        self.state = list()
        for i in range(self.nb_agents):
            new_state.append(get_state(self.model, self.agent[i]))
            reward.append(self.reward.payout(self.state[i], actions[i], new_state[i]))
            done.append(not self.agent[i].active)
            info.append({})
        self.state = new_state

        return self.prepared_state, reward, done, info

    def render(self, mode='ansi'):
        if mode == 'ansi':
            print(self.state[0]["cells"], "\n")
            if not self.model.running:
                self.model.print_standings()
        else:
            return NotImplementedError

    @property
    def prepared_state(self):
        if self.state_preparation_fn:
            return self.state_preparation_fn(self.state)
        else:
            # Todo: Round % 6 counter missing
            # Format: [cells, direction, speed]
            cells = self.state[0]["cells"]
            state = list()
            for i in range(self.nb_agents):
                state.append(np.asarray([
                    *np.asarray(cells).flatten(),
                    Direction[self.state[i]["players"][str(i + 1)]["direction"].upper()].value,
                    self.state[i]["players"][str(i + 1)]["speed"]
                ]))
            return state
