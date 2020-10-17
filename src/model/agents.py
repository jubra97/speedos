from mesa import Agent
from abc import abstractmethod
from itertools import permutations
from src.model.utils import Direction, Action, get_state, arg_maxes, state_to_model, compare_grid_with_cells
import numpy as np


class SpeedAgent(Agent):
    """
    Abstract representation of an Agent in Speed.
    """
    def __init__(self, model, pos, direction, speed=1, active=True):
        """
        :param model: The model that the agent lives in.
        :param pos: The initial position in (x, y)
        :param direction: The initial agent direction as a Direction-object
        :param speed: The initial speed.
        :param active: Whether or not the agent is not eliminated.
        """
        super().__init__(model.next_id(), model)
        self.pos = pos
        self.direction = direction
        self.speed = speed
        self.active = active

        self.action = None
        self.trace = []     # Holds all cells that were visited in the last step
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

        state = get_state(self.model, self)
        self.action = self.act(state)

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
            self.set_inactive()
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
                # remove agent and add the last trace
                self.set_inactive()
                reached_new_pos = False
                break

            # create trace
            # trace gaps occur every 6 rounds if the speed is higher than 2.
            if (self.model.schedule.steps + 1) % 6 != 0 or self.speed < 3 or i == 0 or i == 1:
                self.model.add_agent(AgentTrace(self.model, old_pos, self))
                self.trace.append(new_pos)

        pos = new_pos if reached_new_pos else old_pos
        self.model.grid.move_agent(self, pos)
        # swapped position args since cells has the format (height, width)
        self.model.cells[pos[1], pos[0]] = self.unique_id

        print(f"Agent {self.unique_id} took action {self.action} with speed {self.speed}")
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
        self.elimination_step = self.model.schedule.steps + 1


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
        self.creation_step = self.model.schedule.steps


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
    Agent that chooses random actions.
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

        return arg_maxes(survival.values(), list(survival.keys()))


class ValidationAgent(SpeedAgent):

    def __init__(self, model, pos, direction, speed, active, game, checker_agent):
        super().__init__(model, pos, direction, speed, active)
        self.org_game = game
        self.state_list = []
        self.checker_agent = checker_agent

    def get_action(self, id):
        if self.model.schedule.steps+1 >= len(self.org_game):
            return None
        current_speed = self.org_game[self.model.schedule.steps]["players"][id]["speed"]
        next_speed = self.org_game[self.model.schedule.steps + 1]["players"][id]["speed"]
        current_direction = Direction[
            self.org_game[self.model.schedule.steps]["players"][id]["direction"].upper()].value
        next_direction = Direction[
            self.org_game[self.model.schedule.steps + 1]["players"][id]["direction"].upper()].value
        if current_speed - next_speed == -1:
            return Action.SPEED_UP
        elif current_speed - next_speed == 1:
            return Action.SLOW_DOWN
        elif (current_direction - next_direction) % 4 == 3:
            return Action.TURN_RIGHT
        elif (current_direction - next_direction) % 4 == 1:
            return Action.TURN_LEFT
        elif (current_direction - next_direction) == 0:
            return Action.CHANGE_NOTHING
        else:
            raise AttributeError("Not allowed change of direction!")

    def compare_with_org_game(self, state):
        org_cells = np.array(self.org_game[self.model.schedule.steps]["cells"], dtype="float64")
        current_cells = self.model.cells
        if not (current_cells == org_cells).all():
            print(f"CELLS DO NOT MATCH in Step {self.model.schedule.steps}")

        if self.org_game[self.model.schedule.steps]['players'] != state['players']:
            print("__________")
            print(f"STATES DO NOT MATCH in Step {self.model.schedule.steps}")
            print(f"Org State: {self.org_game[self.model.schedule.steps]['players']}")
            print(f"Sim State: {state['players']}")
            print("__________")

    def act(self, state):
        # TODO: Compare with inactive agents, Is last step compared?
        if state["you"] == self.checker_agent:
            print(f"CHECK FOR STEP {self.model.schedule.steps}")
            self.compare_with_org_game(state)
            compare_grid_with_cells(self.model)
            self.state_list.append(state)
            if self.model.schedule.steps == len(self.org_game)-1:
                for entry in self.org_game:
                    entry.pop("deadline", None)
                    a = np.array(entry["cells"])
                    entry["cells"] = a
                print(self.state_list[0])
                print(self.org_game[0])
                print(self.state_list[0] == self.org_game[0])
        action = self.get_action(str(state["you"]))
        return action
