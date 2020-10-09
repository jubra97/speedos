from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
import numpy as np
from src.model.agents import SpeedAgent, AgentTrace, RandomAgent
from src.model.utils import Direction


class SpeedModel(Model):
    """
    Model of the game "Speed". This class controls the execution of the simulation.
    """
    def __init__(self, width, height, nb_agents, initial_agents_params=None, agent_classes=None, data_collector=None,
                 verbose=True):
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
        self.cells = np.zeros((width, height))

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
                agent = RandomAgent(**agent_params)
            else:
                agent = agent_classes[i](**agent_params)
            self.add_agent(agent)
            self.speed_agents.append(agent)
            self.active_speed_agents.append(agent)

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

    def check_collisions(self):
        """
        Checks every active agent for collisions with traces or other agents. Colliding agents are eliminated.
        :return: None
        """
        # the agent is eliminated if its trace (of the last step) overlaps with any other traces
        for agent in self.active_speed_agents:
            for t in agent.trace:
                cell_contents = self.grid.get_cell_list_contents(t)
                if len(cell_contents) > 1:
                    agent.set_inactive()
                    self.cells[t] = -1

    def check_game_finished(self):
        """
        Checks whether or not the game has finished (every agent is eliminated) and prints the result if finished.
        :return: None
        """
        if len(self.active_speed_agents) == 0:
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
        print("Standings after {} rounds:\n".format(self.schedule.steps + 1), result)

    def add_agent(self, agent):
        """
        Adds an agent to the model.
        :param agent: The agent to add to the model
        :return: None
        """
        self.schedule.add(agent)
        self.grid.place_agent(agent, agent.pos)
        if type(agent) is SpeedAgent:
            self.cells[agent.pos] = agent.unique_id
        elif type(agent) is AgentTrace:
            self.cells[agent.pos] = agent.origin.unique_id

    def remove_agent(self, agent):
        """
        Removes an agent from the model.
        :param agent: The agent to remove from the model
        :return: None
        """
        self.schedule.remove(agent)
        self.grid.remove_agent(agent)
