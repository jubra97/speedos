from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
import numpy as np
from src.model.agents import SpeedAgent, AgentTrace, OneStepSurvivalAgent, AgentTraceCollision, MultiMiniMaxAgent
from src.utils import Direction


class SpeedModel(Model):
    """
    Model of the game "Speed". This class controls the execution of the simulation.
    """
    def __init__(self, width, height, nb_agents, cells=None, initial_agents_params=None, agent_classes=None,
                 data_collector=None, verbose=False):
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
        if initial_agents_params is None:
            initial_agents_params = [{} for i in range(nb_agents)]

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

            if agent_classes is None:
                agent = MultiMiniMaxAgent(**agent_params)
            else:
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
                        agent = AgentTrace(self, (x, y), self.speed_agents[self.cells[y, x]-1])  # get agent based on id
                        self.add_agent(agent)

    def step(self):
        """
        Computes one iteration of the model.
        :return: None
        """
        if self.data_collector:
            self.data_collector.collect(self)
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
