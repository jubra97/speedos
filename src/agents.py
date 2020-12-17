import datetime
import multiprocessing
from itertools import permutations

import numpy as np
import requests
from pynput import keyboard

from src.heuristics import heuristics
from src.model import SpeedAgent
from src.utils import Action, get_state, arg_maxes, state_to_model


class AgentDummy(SpeedAgent):
    """
    Agent dummy for reinforcement learning.
    It doesn't choose and set an action since the Gym environment controls the execution.
    """

    def act(self, state):
        return None

    def step(self):
        pass


class RandomAgent(SpeedAgent):
    """
    Agent that chooses random actions.
    """

    def act(self, state):
        own_id = state["you"]
        own_props = state["players"][str(own_id)]
        possible_actions = list(Action)
        if own_props["speed"] == 1:
            possible_actions.remove(Action.SLOW_DOWN)
        elif own_props["speed"] == 10:
            possible_actions.remove(Action.SPEED_UP)
        return self.random.choice(possible_actions)


class NStepSurvivalAgent(SpeedAgent):
    """
    Agent that calculates all action combinations for the next n (depth) steps and chooses the action that has the
    lowest amount of death paths.
    """
    def __init__(self, model, pos, direction, speed=1, active=True, depth=1, deterministic=False):
        super().__init__(model, pos, direction, speed, active)
        self.depth = depth
        self.survival = None
        self.deterministic = deterministic

    def act(self, state):
        self.survival = dict.fromkeys(list(Action), 0)
        self.deep_search(state, self.depth, None)
        amaxes = arg_maxes(self.survival.values(), list(self.survival.keys()))
        if len(amaxes) == 0:
            amaxes = list(Action)
        if self.deterministic:
            return amaxes[0]
        else:
            return np.random.choice(amaxes)

    def deep_search(self, state, depth, initial_action):
        own_id = state["you"]

        if not state["players"][str(own_id)]["active"]:
            return
        elif depth == 0:
            self.survival[initial_action] += 1
        else:
            model = state_to_model(state)
            nb_active_agents = len(model.active_speed_agents)
            action_permutations = list(permutations(list(Action), nb_active_agents))
            for action_permutation in action_permutations:
                own_agent = model.get_agent_by_id(own_id)
                for idx, agent in enumerate(model.active_speed_agents):
                    agent.action = action_permutation[idx]
                model.step()
                new_state = get_state(model, own_agent, self.deadline)
                # recursion
                if initial_action is None:
                    self.deep_search(new_state, depth - 1, own_agent.action)
                else:
                    self.deep_search(new_state, depth - 1, initial_action)
                model = state_to_model(state)


class HumanAgent(SpeedAgent):

    def act(self, state):
        with keyboard.Events() as events:
            # Block for as much as possible
            input_key = events.get(1000000).key

        if input_key == keyboard.KeyCode.from_char('w'):
            return Action.SPEED_UP
        elif input_key == keyboard.KeyCode.from_char('s'):
            return Action.SLOW_DOWN
        elif input_key == keyboard.KeyCode.from_char('a'):
            return Action.TURN_LEFT
        elif input_key == keyboard.KeyCode.from_char('d'):
            return Action.TURN_RIGHT
        else:
            return Action.CHANGE_NOTHING


class MultiMiniMaxAgent(SpeedAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm
    """
    def __init__(self, model, pos, direction, speed=1, active=True, base_depth=2, use_voronoi=True):
        super().__init__(model, pos, direction, speed, active)
        self.base_depth = base_depth
        self.use_voronoi = use_voronoi

    def act(self, state):
        depth = self.base_depth
        # depth = self.base_depth + model.nb_agents - len(model.active_speed_agents)
        action = heuristics.multi_minimax(depth, state, use_voronoi=self.use_voronoi)

        return action


class MultiMiniMaxDeadlineAwareAgent(SpeedAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm before deadline
    """
    def __init__(self, model, pos, direction, speed=1, active=True, base_depth=2, use_voronoi=True):
        super().__init__(model, pos, direction, speed, active)
        self.base_depth = base_depth
        self.use_voronoi = use_voronoi
        self.super_pruning = False

    def act(self, state):
        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=heuristics.multi_minimax_depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state, self.super_pruning, self.use_voronoi))
        p.start()
        send_time = 3
        deadline = datetime.datetime.strptime(state["deadline"], "%Y-%m-%dT%H:%M:%SZ")
        response = requests.get("https://msoll.de/spe_ed_time")
        server_time = datetime.datetime.strptime(response.json()["time"], "%Y-%m-%dT%H:%M:%SZ")
        av_time = (deadline - server_time).total_seconds() - send_time
        p.join(av_time)

        # If thread is active
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()

        print(f"reached_depth: {reached_depth.value}")
        return Action(move.value)



