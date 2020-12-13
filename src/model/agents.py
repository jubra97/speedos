from mesa import Agent
from mesa import Model
from abc import abstractmethod
from itertools import permutations
from src.utils import Direction, Action, get_state, arg_maxes, state_to_model, model_to_json, speed_one_voronoi
from src.utils import Direction, Action, get_state, arg_maxes, state_to_model
from src.heuristics import heuristics
import numpy as np
#from pynput import keyboard
#import matplotlib.pyplot as plt
import multiprocessing
import datetime
import random


class SpeedAgent(Agent):
    """
    Abstract representation of an Agent in Speed.
    """

    def __init__(self, model, pos, direction, speed=1, active=True, trace=None):
        """
        :param model: The model that the agent lives in.
        :param pos: The initial position in (x, y)
        :param direction: The initial agent direction as a Direction-object
        :param speed: The initial speed.
        :param active: Whether or not the agent is not eliminated.
        """
        if model is None:
            # use an empty model if an agent is used to play against an online game
            model = Model()
            super().__init__(None, model)
        else:
            super().__init__(model.next_id(), model)
        self.pos = pos
        self.direction = direction
        self.speed = speed
        self.active = active
        self.deadline = None
        # Holds all cells that were visited in the last step
        if trace is None:
            self.trace = []
        else:
            self.trace = trace

        self.action = None
        self.elimination_step = -1  # Saves the step that the agent was eliminated in (-1 if still active)

    @abstractmethod
    def act(self, state):
        """
        Chooses an action - should be overwritten by an agent implementation.
        :return: Action
        """
        return NotImplementedError

    def step(self):
        """
        Fetches the current state and sets the action.
        :return: None
        """
        if not self.active:
            return

        # set deadline in agent because every agent has 10 seconds of time.
        acceptable_computing_time = datetime.timedelta(seconds=9.8 + random.uniform(-0.3, 0.3))
        self.deadline = datetime.datetime.utcnow() + acceptable_computing_time

        state = get_state(self.model, self, self.deadline)
        self.action = self.act(state)
        if datetime.datetime.utcnow() > self.deadline:
            print(f"Agent {self} exceeded Deadline by {datetime.datetime.utcnow() - self.deadline}!")
            self.set_inactive()

    def advance(self):
        """
        Executes the selected action and moves the agent.
        :return: None
        """
        if not self.active:
            return

        if self.action == Action.TURN_LEFT:
            self.direction = Direction((self.direction.value - 1) % 4)
        elif self.action == Action.TURN_RIGHT:
            self.direction = Direction((self.direction.value + 1) % 4)
        elif self.action == Action.SLOW_DOWN:
            self.speed -= 1
        elif self.action == Action.SPEED_UP:
            self.speed += 1

        self.move()

    def move(self):
        """
        Move the agent according to its direction and speed and creates agent traces.
        :return: None
        """
        if not self.valid_speed():
            self.trace = []
            self.set_inactive()
            self.elimination_step += 1
            return

        # empty the trace
        self.trace = []

        # init new pos
        new_x = self.pos[0]
        new_y = self.pos[1]
        old_pos = new_pos = (new_x, new_y)
        reached_new_pos = True

        # visit all cells that are within "self.speed"
        for i in range(self.speed):
            # update position
            if self.direction == Direction.UP:
                new_y -= 1
            elif self.direction == Direction.DOWN:
                new_y += 1
            elif self.direction == Direction.LEFT:
                new_x -= 1
            elif self.direction == Direction.RIGHT:
                new_x += 1
            old_pos = new_pos
            new_pos = (new_x, new_y)
            # check borders and speed
            if self.model.grid.out_of_bounds(new_pos):
                # add trace at last in front of bound if speed is slow
                if (self.model.schedule.steps + 1) % 6 != 0 or i == 1 or i == 0:
                    self.model.add_agent(AgentTrace(self.model, old_pos, self))
                # remove agent from grid
                self.model.grid.remove_agent(self)
                # set pos for matching with original game
                self.pos = (new_x, new_y)
                self.set_inactive()
                self.elimination_step += 1
                reached_new_pos = False
                break

            # create trace
            # trace gaps occur every 6 rounds if the speed is higher than 2.
            if (self.model.schedule.steps + 1) % 6 != 0 or self.speed < 3 or i == 1 or i == 0:
                self.model.add_agent(AgentTrace(self.model, old_pos, self))
                self.trace.append(new_pos)

        # only move agent if new pos is in bounds
        pos = new_pos
        if reached_new_pos:
            self.model.grid.move_agent(self, pos)
            # swapped position args since cells has the format (height, width)
            self.model.cells[pos[1], pos[0]] = self.unique_id

    def valid_speed(self):
        return 1 <= self.speed <= 10

    def set_inactive(self):
        """
        Eliminates the agent from the round.
        :return: None
        """
        self.active = False
        if self in self.model.active_speed_agents:
            self.model.active_speed_agents.remove(self)
        self.elimination_step = self.model.schedule.steps


class AgentDummy(SpeedAgent):
    """
    Agent dummy for reinforcement learning.
    It doesn't choose and set an action since the Gym environment controls the execution.
    """

    def act(self, state):
        return None

    def step(self):
        pass


class AgentTrace(Agent):
    """
    Static trace element of an Agent that occupies one cell.
    """

    def __init__(self, model, pos, origin):
        """
        :param model: The model that the trace exists in.
        :param pos: The position of the trace element in (x, y)
        :param origin: The SpeedAgent Object that produced the trace
        """
        super().__init__(model.next_id(), model)
        self.pos = pos
        self.origin = origin


class AgentTraceCollision(AgentTrace):
    """
    Static Agent to mark a Collision. Agent_id is always -1.
    """

    def __init__(self, model, pos):
        super().__init__(model, pos, None)


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


class OneStepSurvivalAgent(SpeedAgent):
    """
    Agent that calculates the next step and chooses an action where he survives.
    """
    def act(self, state):
        own_id = state["you"]
        survival = dict.fromkeys(list(Action), 0)
        model = state_to_model(state)

        nb_active_agents = len(model.active_speed_agents)
        action_permutations = list(permutations(list(Action), nb_active_agents))
        for action_permutation in action_permutations:
            own_agent = model.get_agent_by_id(own_id)
            for idx, agent in enumerate(model.active_speed_agents):
                agent.action = action_permutation[idx]

            model.step()
            if own_agent.active:
                survival[own_agent.action] += 1
            model = state_to_model(state)

        amaxes = arg_maxes(survival.values(), list(survival.keys()))
        if len(amaxes) == 0:
            amaxes = list(Action)
        return np.random.choice(amaxes)


class NStepSurvivalAgent(SpeedAgent):
    """
    Agent that calculates the next steps and chooses an action where he survives.
    """
    def __init__(self, model, pos, direction, speed=1, active=True, depth=2):
        super().__init__(model, pos, direction, speed, active)
        self.depth = depth
        self.survival = None

    def act(self, state):
        self.survival = dict.fromkeys(list(Action), 0)
        self.deep_search(state, self.depth, None)
        amaxes = arg_maxes(self.survival.values(), list(self.survival.keys()))
        if len(amaxes) == 0:
            amaxes = list(Action)
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
        from pynput import keyboard

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
        av_time = (deadline - datetime.datetime.utcnow()).total_seconds() - send_time
        p.join(av_time)

        # If thread is active
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()

        return Action(move.value)



