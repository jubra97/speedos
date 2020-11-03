from enum import Enum
import numpy as np


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self):
        return self.name.lower()


class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    SLOW_DOWN = 2
    SPEED_UP = 3
    CHANGE_NOTHING = 4

    def __str__(self):
        return self.name.lower()


def agent_to_json(agent):
    x, y = agent.pos
    return {
        "x": x,
        "y": y,
        "direction": str(agent.direction),
        "speed": agent.speed,
        "active": agent.active
    }


def model_to_json(model):
    players = dict()
    for agent in model.speed_agents:
        players[str(agent.unique_id)] = agent_to_json(agent)

    return {
        "width": model.width,
        "height": model.height,
        "cells": model.cells.copy(),
        "players": players,
        "running": model.running
    }


def get_state(model, agent):
    state = model_to_json(model)
    state["you"] = agent.unique_id
    return state


def arg_maxes(arr, indices=None):
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


def state_to_model(state, initialize_cells=False, agent_classes=None, additional_params=None):
    width = state["width"]
    height = state["height"]
    nb_agents = len(state["players"])
    initial_params = []
    for i, values in enumerate(state["players"].values()):
        initial_params.append({
            "pos": (values["x"], values["y"]),
            "direction": Direction[values["direction"].upper()],
            "speed": values["speed"],
            "active": values["active"]
        })
        if additional_params is not None:
            initial_params[i] = {**initial_params[i], **additional_params[i]}
    # TODO: doesnt work with global import, cyclic import?
    from src.model.model import SpeedModel
    from src.model.agents import AgentDummy
    if agent_classes is None:
        agent_classes = [AgentDummy for i in range(nb_agents)]
    model = SpeedModel(width, height, nb_agents, state["cells"] if not initialize_cells else None, initial_params,
                       agent_classes)
    return model


def evaluate_position(model, agent):
    if not agent.active:
        return -1
    else:
        return 1
