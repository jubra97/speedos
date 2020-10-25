from gym import Env, spaces
from src.model.model import SpeedModel
from src.model.agents import AgentDummy
from src.utils import Action
from src.utils import get_state
from src.rl.gym.rewards import LongSurvivalReward, WinLossReward


class SingleAgentSpeedEnv(Env):
    """
    A Gym environment of Speed with only the learning agent and no opponents.
    An optimal agent learns to survive as long as possible (fill as many cells as possible before being eliminated)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width, height, observer, reward=LongSurvivalReward()):
        self.observer = observer
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = observer.observation_space
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
        self.model = SpeedModel(self.width, self.height, nb_agents=1, agent_classes=[AgentDummy], verbose=False)
        self.seed()
        self.agent = self.model.speed_agents[0]
        self.state = get_state(self.model, self.agent)

        return self.observer.prepared_state(self.state, self.model.schedule.steps)

    def step(self, action):
        self.agent.action = Action(action)
        self.model.step()

        new_state = get_state(self.model, self.agent)
        done = not self.agent.active
        reward = self.reward.payout(self.state, action, new_state, done, new_state["you"])
        info = {}
        self.state = new_state

        return self.observer.prepared_state(self.state, self.model.schedule.steps), reward, done, info

    def render(self, mode='ansi'):
        if mode == 'ansi':
            print(self.agent.action, "\n")
            print(self.model.cells, "\n")
            if not self.agent.active:
                self.model.print_standings()
        else:
            return NotImplementedError

    def close(self):
        pass


class HeuristicsSpeedEnv(SingleAgentSpeedEnv):
    """
    A Gym environment of Speed with a given set of static heuristic opponents.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width, height, observer, agent_classes, reward=WinLossReward()):
        self.agent_classes = agent_classes
        super(HeuristicsSpeedEnv, self).__init__(width, height, observer, reward)

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=self.observer.nb_agents,
                                agent_classes=self.agent_classes, verbose=False)
        self.seed()
        self.agent = self.model.speed_agents[0]
        self.state = get_state(self.model, self.agent)

        return self.observer.prepared_state(self.state, self.model.schedule.steps)

    def step(self, action):
        next_state, reward, done, info = super(HeuristicsSpeedEnv, self).step(action)
        done = not self.model.running
        return next_state, reward, done, info


class SelfPlaySpeedEnv(SingleAgentSpeedEnv):
    """
    A Gym environment of Speed that lets the agent play against itself.
    The step()-method expects an array of actions (one for each agent instance) and returns lists of new states,
    rewards, dones and infos (with new_state[i], reward[i], done[i] and info[i]) being the transition for agent i.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, width, height, observer, reward=WinLossReward()):
        self.nb_agents = observer.nb_agents
        super(SelfPlaySpeedEnv, self).__init__(width, height, observer, reward)

    def reset(self):
        # Use an AgentDummy since the Gym environment controls the execution.
        self.model = SpeedModel(self.width, self.height, nb_agents=self.nb_agents,
                                agent_classes=[AgentDummy for _ in range(self.nb_agents)], verbose=False)
        self.seed()
        self.agent = self.model.speed_agents
        self.state = list(map(get_state, [self.model for _ in range(self.nb_agents)], self.agent))

        return [self.observer.prepared_state(self.state[i], self.state[i]["you"], self.model.schedule.steps)
                for i in range(self.nb_agents)]

    def step(self, actions):
        for i in range(self.nb_agents):
            self.agent[i].action = Action(actions[i])
        self.model.step()

        transitions = list()
        done = not self.model.running
        for i in range(self.nb_agents):
            new_state = get_state(self.model, self.agent[i])
            reward = self.reward.payout(self.state[i], actions[i], new_state, done, new_state["you"])
            # agents that have already been eliminated before this step should not be used for learning anymore
            info = {"skip": not self.agent[i].active and self.agent[i].elimination_step == self.model.schedule.steps}

            self.state[i] = new_state
            transitions.append((
                self.observer.prepared_state(self.state[i], self.state[i]["you"], self.model.schedule.steps),
                reward,
                done,
                info
            ))

        return transitions

    def render(self, mode='ansi'):
        if mode == 'ansi':
            print(self.model.cells, "\n")
            if not self.model.running:
                self.model.print_standings()
        else:
            return NotImplementedError
