import copy
import hashlib
import json
from enum import Enum


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


def out_of_bounds(cell_size, pos) -> bool:
    x, y = pos
    return x < 0 or x >= cell_size[1] or y < 0 or y >= cell_size[0]


def hash_state(state):
    val = ""
    for l in state["cells"]:
        val += "".join(map(str, l))
    return int(hashlib.sha1(val.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def agent_to_json(agent, trace_aware=False):
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


def model_to_json(model, trace_aware=False):
    players = dict()
    for agent in model.speed_agents:
        players[str(agent.unique_id)] = agent_to_json(agent, trace_aware)

    return {
        "width": model.width,
        "height": model.height,
        "cells": model.cells.copy(),
        "players": players,
        "running": model.running
    }


def get_state(model, agent, deadline=None):
    state = model_to_json(model)
    state["you"] = agent.unique_id
    if deadline is not None:
        state["deadline"] = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
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


def state_to_model(state, initialize_cells=False, agent_classes=None, additional_params=None, trace_aware=False):
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
    # TODO: doesnt work with global import, cyclic import?
    from src.model import SpeedModel
    from src.agents import AgentDummy
    if agent_classes is None:
        agent_classes = [AgentDummy for i in range(nb_agents)]
    model = SpeedModel(width, height, nb_agents, agent_classes, initial_agents_params=initial_params,
                       cells=state["cells"] if not initialize_cells else None)
    agents_to_remove = []
    for agent in model.active_speed_agents:
        if not agent.active:
            agents_to_remove.append(agent)
    for agent in agents_to_remove:
        model.active_speed_agents.remove(agent)
    return model


def json_to_history(path_to_json, output_path, horizontal=False):
    """
    Copy the output of this function to the excel sheet misc/Speed_History.xlsx
    to get conditional formatting.
    :param path_to_json:
    :param output_path: should end on .tsv or .txt
    :param horizontal: vertical or horizintal data format for output
    :return:
    """
    with open(path_to_json, encoding="utf-8") as f:
        data = json.load(f)

    outfile = open(output_path, "w+", encoding="utf-8")

    if horizontal:
        for i in range(0, data[0]["height"]):
            for r in range(0, len(data)):
                row = data[r]["cells"][i]
                outfile.write("\t".join((map(lambda x: str(x), row))))
                outfile.write("\t-\t")
            outfile.write("\n")
    else:
        for round in data:
            outrows = []
            for row in round["cells"]:
                outrows.append("\t".join(map(lambda x: str(x), row)))
            outfile.write("\n".join(outrows))
            outfile.write("\n-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n")

    outfile.close()
