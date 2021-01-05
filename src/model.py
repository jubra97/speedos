import copy
import datetime
import json
import os
import random
from abc import abstractmethod

import numpy as np
from mesa import Agent
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation

from src.utils import Direction, Action, get_state
from src.utils import model_to_json


class SpeedModel(Model):
    """
    Model of the game "Speed". This class controls the execution of the simulation.
    """

    def __init__(self, width, height, nb_agents, agent_classes, initial_agents_params=None, cells=None,
                 data_collector=None, verbose=False, save=False):
        """
        :param initial_agents_params: A list of dictionaries containing initialization parameters for agents
        that should be initialized at the start of the simulation
        :param data_collector:
        """
        super().__init__()
        self.data_collector = data_collector
        self.width = width
        self.height = height
        self.nb_agents = nb_agents
        self.verbose = verbose
        self.save = save
        if self.save:
            self.history = []
        if initial_agents_params is None:
            initial_agents_params = [{} for i in range(nb_agents)]
        else:
            initial_agents_params = copy.deepcopy(initial_agents_params)

        self.schedule = SimultaneousActivation(self)

        self.grid = MultiGrid(width, height, True)
        # width and height are swapped since height is rows and width is columns
        # an alternative to this representation would be to transpose cells everytime it is exposed
        # but that could be inefficient
        self.cells = np.zeros((height, width), dtype="int")

        # Init initial agents
        self.speed_agents = []
        self.active_speed_agents = []
        for i in range(nb_agents):
            agent_params = initial_agents_params[i]
            agent_params["model"] = self
            # set to random position/direction if no position/direction is given
            if "pos" not in agent_params:
                agent_params["pos"] = self.random.choice(list(self.grid.empties))
            if "direction" not in agent_params:
                agent_params["direction"] = self.random.choice(list(Direction))

            agent = agent_classes[i](**agent_params)

            # don't add agent to grid/cells if its out of bounds. But add it to the scheduler.
            if self.grid.out_of_bounds(agent_params["pos"]):
                self.schedule.add(agent)
                self.speed_agents.append(agent)
            else:
                self.add_agent(agent)
                self.speed_agents.append(agent)
                self.active_speed_agents.append(agent)

        if cells is not None:
            self.init_cells_and_grid(cells)

    def init_cells_and_grid(self, cells):
        self.cells = np.array(cells)
        # add traces to grid
        for y in range(self.cells.shape[0]):
            for x in range(self.cells.shape[1]):
                # cell is occupied by a collision
                if self.cells[y, x] == -1:
                    agent = AgentTraceCollision(self, (x, y))
                    self.add_agent(agent)
                # cell is occupied by head or trace
                elif self.cells[y, x] != 0:
                    # head of the agent is not already a entry in self.grid
                    if len(self.grid.get_cell_list_contents((x, y))) == 0:
                        # add trace
                        try:
                            agent = AgentTrace(self, (x, y),
                                               self.speed_agents[self.cells[y, x] - 1])  # get agent based on id
                            self.add_agent(agent)
                        except:
                            print(self.cells)
                            print(self.speed_agents)
                            print(self.cells[y, x] - 1)

    def step(self):
        """
        Computes one iteration of the model.
        :return: None
        """
        if self.data_collector:
            self.data_collector.collect(self)
        if self.save:
            self.history.append(copy.deepcopy(model_to_json(self)))
        self.schedule.step()
        self.check_collisions()
        self.check_game_finished()

    def step_specific_agent(self, agent):
        """
        Only steps one specific agent. This is only for specific applications (e.g. Multi-Minimax).
        Don't use this method if not necessary since it doesn't increment all model parts (e.g. time).
        :return: None
        """
        agent.step()
        agent.advance()
        self.check_collisions()
        self.check_game_finished()

    def check_collisions(self):
        """
        Checks every active agent for collisions with traces or other agents. Colliding agents are eliminated.
        :return: None
        """
        agents_to_set_inactive = []
        for agent in self.speed_agents:
            for t in agent.trace:
                cell_contents = self.grid.get_cell_list_contents(t)
                if len(cell_contents) > 1:
                    if agent not in agents_to_set_inactive:
                        agents_to_set_inactive.append(agent)
                    self.add_agent(AgentTraceCollision(self, t))

        for agent in agents_to_set_inactive:
            agent.set_inactive()

    def check_game_finished(self):
        """
        Checks whether or not the game has finished (every agent is eliminated) and prints the result if finished.
        :return: None
        """
        if len(self.active_speed_agents) <= 1:
            self.running = False
            if self.verbose:
                self.print_standings()
            if self.save:
                self.history.append(copy.deepcopy(model_to_json(self)))
                path = os.path.abspath("") + "/res/simulatedGames/"
                for entry in self.history:
                    entry["cells"] = entry["cells"].tolist()
                with open(path + datetime.datetime.now().strftime("%d-%m-%y__%H-%M-%S-%f") + ".json", "w") as f:
                    json.dump(self.history, f, indent=4)

    def print_standings(self):
        """
        Print the amount of survived steps of each agent.
        :return:
        """
        result = list(map(
            lambda agent: {"ID: ": agent.unique_id, "Survived Steps: ": agent.elimination_step},
            self.speed_agents
        ))
        if len(self.active_speed_agents) == 1:
            print('Winning Agent: ' + str(self.active_speed_agents[0].unique_id))
        else:
            print('Draw')
        print("Standings after {} rounds:\n".format(self.schedule.steps), result)

    def add_agent(self, agent):
        """
        Adds an agent to the model.
        :param agent: The agent to add to the model
        :return: None
        """
        self.schedule.add(agent)
        self.grid.place_agent(agent, agent.pos)
        # swapped position args since cells has the format (height, width)
        pos = (agent.pos[1], agent.pos[0])
        if isinstance(agent, SpeedAgent):
            self.cells[pos] = agent.unique_id
        elif type(agent) is AgentTraceCollision:
            self.cells[pos] = -1
        elif type(agent) is AgentTrace:
            self.cells[pos] = agent.origin.unique_id

    def remove_agent(self, agent):
        """
        Removes an agent from the model.
        :param agent: The agent to remove from the model
        :return: None
        """
        if agent in self.schedule.agents:
            self.schedule.remove(agent)
            self.grid.remove_agent(agent)

    def get_agent_by_id(self, unique_id):
        for agent in self.speed_agents:
            if agent.unique_id == unique_id:
                return agent
        return None


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
            if self.trace[-1] != new_pos:
                self.trace.append(new_pos)
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