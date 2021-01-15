import copy
from enum import Enum
import numpy as np


class Direction(Enum):
    """
    Class to describe the possible directions in the game as Enum.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self):
        return self.name.lower()


class Action(Enum):
    """
    Class to describe the possible actions in the game as Enum.
    """
    TURN_LEFT = 0
    TURN_RIGHT = 1
    SLOW_DOWN = 2
    SPEED_UP = 3
    CHANGE_NOTHING = 4

    def __str__(self):
        return self.name.lower()


def out_of_bounds(cell_size, pos) -> bool:
    """
    Checks whether a position is out of the game board.
    :param cell_size: Board size as array
    :param pos: tuple (x, y)
    :return: bool
    """
    x, y = pos
    return x < 0 or x >= cell_size[1] or y < 0 or y >= cell_size[0]


def agent_to_json(agent, trace_aware=False):
    """
    Converts agent to JSON string
    :param agent: the agent
    :param trace_aware: include traces
    :return: json string with x, y, direction, speed, active
    """
    x, y = agent.pos
    agent_json = {
        "x": x,
        "y": y,
        "direction": str(agent.direction),
        "speed": agent.speed,
        "active": agent.active
    }
    if trace_aware:
        agent_json["trace"] = copy.deepcopy(agent.trace)
    return agent_json


def model_to_json(model, trace_aware=False, step=False):
    """
    Converts the model to JSON string
    :param model: the model
    :param trace_aware: include traces
    :param step: current game step
    :return: JSON string with width, height, cells, players, running, [step]
    """
    players = dict()
    for agent in model.speed_agents:
        players[str(agent.unique_id)] = agent_to_json(agent, trace_aware)

    state = {
        "width": model.width,
        "height": model.height,
        "cells": model.cells.copy(),
        "players": players,
        "running": model.running
    }
    if step:
        state["step"] = model.schedule.steps
    return state


def get_state(model, agent, deadline=None, step=False):
    """
    Gets model state and sets id for current player
    :param model: the model
    :param agent: current agent
    :param deadline: optional deadline for move
    :param step: current game step
    :return: JSON string of state
    """
    state = model_to_json(model, step=step)
    state["you"] = agent.unique_id
    if deadline is not None:
        state["deadline"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
    return state


def arg_maxes(arr, indices=None):
    """
    Get indices of all maximal values in array
    :param arr: the array
    :param indices: optional permutation on indices
    :return: array with indices of maxima
    """
    if len(arr) == 0:
        return []

    maxes = []
    maximum = max(arr)
    for idx, el in enumerate(arr):
        if el == maximum:
            if indices:
                maxes.append(indices[idx])
            else:
                maxes.append(idx)
    return maxes


def state_to_model(state, initialize_cells=False, agent_classes=None, additional_params=None, trace_aware=False):
    """
    Convert JSON state to model
    :param state: JSON string
    :param initialize_cells: initialize cells
    :param agent_classes: agent classes
    :param additional_params: additional params
    :param trace_aware: include traces
    :return: model
    """
    # import here to avoid cyclic imports
    from src.core.model import SpeedModel
    from src.core.agents import DummyAgent
    width = state["width"]
    height = state["height"]
    nb_agents = len(state["players"])
    initial_params = []
    for i, values in enumerate(state["players"].values()):
        initial_params.append({
            "pos": (values["x"], values["y"]),
            "direction": Direction[values["direction"].upper()],
            "speed": values["speed"],
            "active": values["active"],
        })
        if trace_aware:
            initial_params[i]["trace"] = copy.deepcopy(values["trace"])
        if additional_params is not None:
            initial_params[i] = {**initial_params[i], **additional_params[i]}

    if agent_classes is None:
        agent_classes = [DummyAgent for i in range(nb_agents)]
    model = SpeedModel(width, height, nb_agents, agent_classes, initial_agents_params=initial_params,
                       cells=state["cells"] if not initialize_cells else None)
    agents_to_remove = []
    for agent in model.active_speed_agents:
        if not agent.active:
            agents_to_remove.append(agent)
    for agent in agents_to_remove:
        model.active_speed_agents.remove(agent)

    if "step" in state.keys():
        model.schedule.steps = state["step"]
    return model


def reduce_state_to_sliding_window(state, distance_to_next_opponent, min_sliding_window_size,
                                   sliding_window_size_offset=3):
    """
    Reduces the state to a smaller window around the agent
    :param state: the tstae
    :param distance_to_next_opponent: distance to next opponent to include
    :param min_sliding_window_size: minimum window size
    :param sliding_window_size_offset: offset
    :return: reduced state
    """
    cells = np.array(state["cells"])
    pos = (state["players"][str(state["you"])]["y"], state["players"][str(state["you"])]["x"])

    if distance_to_next_opponent > min_sliding_window_size:
        sliding_window_size = int(distance_to_next_opponent) + sliding_window_size_offset
    else:
        sliding_window_size = min_sliding_window_size

    upper_bound = pos[0] - sliding_window_size if (pos[0] - sliding_window_size > 0) else 0
    left_bound = pos[1] - sliding_window_size if (pos[1] - sliding_window_size > 0) else 0
    new_cells = cells[upper_bound: pos[0] + sliding_window_size + 1, left_bound: pos[1] + sliding_window_size + 1]
    state["height"] = new_cells.shape[0]
    state["width"] = new_cells.shape[1]
    players_to_remove = []
    for player_number in state["players"]:
        player = state["players"][player_number]
        if player["y"] < pos[0] - sliding_window_size or player["y"] > pos[0] + sliding_window_size or \
                player["x"] < pos[1] - sliding_window_size or player["x"] > pos[1] + sliding_window_size:
            players_to_remove.append(player_number)
        else:
            player["y"] = player["y"] - upper_bound
            player["x"] = player["x"] - left_bound
    for rm_player in players_to_remove:
        del state["players"][rm_player]
    players = {}
    player_numbers = []
    for i, player_number in enumerate(state["players"], 1):
        players[f"{i}"] = state["players"][player_number]
        new_cells[new_cells == int(player_number)] = i
        player_numbers.append(i)
        if player_number == str(state["you"]):
            state["you"] = i
    state["players"] = players

    players_in_cells = np.unique(new_cells).tolist()
    players_in_cells.remove(0)
    for player_in_cell in players_in_cells:
        if player_in_cell not in player_numbers:
            new_cells[new_cells == player_in_cell] = -1

    state["cells"] = new_cells.tolist()

    return state
