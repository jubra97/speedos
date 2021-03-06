import json
import os
import unittest

import numpy as np
from mesa import Model

from src.core.agents import DummyAgent
from src.core.utils import Direction, Action, state_to_model, get_state


@unittest.skip("json files")
class TestModelValidity(unittest.TestCase):

    def setUp(self):
        self.original_games_path = os.path.abspath("../..") + "/res/originalGames/"
        self.test_games = os.listdir(self.original_games_path)
        self.model = Model()
        self.maxDiff = None

    def test_default_params(self):
        for game in self.test_games:
            # uncomment this for debugging to find out which game has failed
            # print(f"Checking Game: {game}")
            path_to_game = self.original_games_path + game
            with open(path_to_game, "r") as file:
                game = json.load(file)

            game = self.remove_duplicates(game)

            initial_state = game[0]
            model = state_to_model(initial_state, False, [DummyAgent for _ in range(len(initial_state["players"]))])
            own_agent = model.get_agent_by_id(1)
            while model.running:
                state = get_state(model, own_agent)

                # execute unit tests
                self.compare_states(game, model, state)
                self.compare_grid_with_cells(model)

                for agent in model.speed_agents:
                    agent.action = self.get_action(game, model, str(agent.unique_id))
                    if agent.action == "set_inactive" and agent.active:
                        agent.set_inactive()
                model.step()

    def compare_states(self, org_game, model, state):
        org_cells = np.array(org_game[model.schedule.steps]["cells"], dtype="float64")
        current_cells = model.cells
        org_state = (org_game[model.schedule.steps]["players"])
        for key, value in org_state.items():
            # remove "name" key in last step to match between org and sim states
            if 'name' in value.keys():
                org_state[key].pop("name")

        # compare cells
        self.assertTrue((current_cells == org_cells).all())
        # compare players
        self.assertDictEqual(org_state, state['players'])

    def compare_grid_with_cells(self, model):
        """
        Checks for differences between the cell and grid representation.
        :param model: The model to be checked
        :return:
        """
        from src.core.model import AgentTrace, AgentTraceCollision
        grid_as_np_array = np.empty((model.height, model.width), dtype="int")
        for entry, x, y in model.grid.coord_iter():
            if len(entry) == 0:
                grid_as_np_array[y, x] = 0
            elif len(entry) == 1:
                agent = next(iter(entry))
                if type(agent) is AgentTraceCollision:
                    grid_as_np_array[y, x] = -1
                elif isinstance(agent, AgentTrace):
                    grid_as_np_array[y, x] = agent.origin.unique_id
                else:
                    grid_as_np_array[y, x] = agent.unique_id
            else:
                if any(type(agent) is AgentTraceCollision for agent in entry):
                    grid_as_np_array[y, x] = -1
                else:
                    raise AssertionError('There has to be an AgentTraceCollision instance '
                                         'on a cells with multiple agents')

        self.assertTrue((model.cells == grid_as_np_array).all())

    @staticmethod
    def get_action(org_game, model, agent_id):
        """
        Compute action based on current and next state of the original game.
        """
        if model.schedule.steps + 1 >= len(org_game):
            return None
        current_speed = org_game[model.schedule.steps]["players"][agent_id]["speed"]
        next_speed = org_game[model.schedule.steps + 1]["players"][agent_id]["speed"]
        current_direction = Direction[
            org_game[model.schedule.steps]["players"][agent_id]["direction"].upper()].value
        next_direction = Direction[
            org_game[model.schedule.steps + 1]["players"][agent_id]["direction"].upper()].value
        # In the real game the the agent sometimes doesn't move and gets inactive.
        # This behavior is not implemented in the model but should not let the tests fail.
        if org_game[model.schedule.steps]["players"][agent_id]["x"] == \
                org_game[model.schedule.steps + 1]["players"][agent_id]["x"] and \
                org_game[model.schedule.steps]["players"][agent_id]["y"] == \
                org_game[model.schedule.steps + 1]["players"][agent_id]["y"] and \
                org_game[model.schedule.steps]["players"][agent_id]["speed"] == \
                org_game[model.schedule.steps + 1]["players"][agent_id]["speed"]:
            return "set_inactive"
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

    @staticmethod
    def remove_duplicates(game):
        rounds_to_remove = []
        for i in range(len(game) - 2):
            if game[i] == game[i + 1]:
                rounds_to_remove.append(i)
        for r in reversed(rounds_to_remove):
            game.pop(r)
        return game
