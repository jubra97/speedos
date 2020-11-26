import copy
from enum import Enum
import numpy as np
import json


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
    from src.model.model import SpeedModel
    from src.model.agents import AgentDummy
    if agent_classes is None:
        agent_classes = [AgentDummy for i in range(nb_agents)]
    model = SpeedModel(width, height, nb_agents, state["cells"] if not initialize_cells else None, initial_params,
                       agent_classes)
    agents_to_remove = []
    for agent in model.active_speed_agents:
        if not agent.active:
            agents_to_remove.append(agent)
    for agent in agents_to_remove:
        model.active_speed_agents.remove(agent)
    return model


def evaluate_position(model, agent):
    if not agent.active:
        return -1
    else:
        return 1


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


def reachable_cells(model, agent):
    marker = 10
    reachable_cells_counter = 0
    cells = model.cells.copy()
    width, height = model.width, model.height

    init_particles = surrounding_cells(agent.pos, width, height)
    particles = []
    for particle in init_particles:
        if cells[particle[1], particle[0]] == 0:
            cells[particle[1], particle[0]] = marker
            particles.append(particle)
            reachable_cells_counter += 1

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            surrounding = surrounding_cells(particle, width, height)
            for cell in surrounding:
                if cells[cell[1], cell[0]] == 0:
                    cells[cell[1], cell[0]] = marker
                    new_particles.append(cell)
                    reachable_cells_counter += 1
        particles = new_particles
    return reachable_cells_counter, cells


def surrounding_cells(position, width, height):
    cells = []
    x, y = position
    for d_x, d_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if 0 <= x + d_x < width and 0 <= y + d_y < height:
            cells.append((x + d_x, y + d_y))
    return cells


class Particle:

    def __init__(self, position, direction, speed, timestamp):
        self.position = position
        self.direction = direction
        self.speed = speed
        self.timestamp = timestamp


def reachable_cells_with_timestamp(model, agent):
    cells = model.cells
    particle_cells = np.empty(cells.shape, dtype=object)
    # TODO: Add cells array that holds all trace-positions with the lowest timestamp
    #  (so we can see if an agent can cut off another agent)

    init_particle = Particle(agent.pos, agent.direction, agent.speed, model.schedule.steps)
    particles = get_new_particles(cells, particle_cells, init_particle)

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            new_particles.extend(get_new_particles(cells, particle_cells, particle))

        particles = new_particles
    return particle_cells


def get_new_particles(cells, particle_cells, particle):
    new_particles = []
    for action in list(Action):
        trace, new_particle = get_action_trace(particle.position, particle.direction, particle.speed, action,
                                               particle.timestamp, cells.shape)
        collision = False
        for t in trace:
            if cells[t[1], t[0]] != 0:
                collision = True

        # TODO: Maybe add a timestamp-difference threshold to avoid throwing away most slow particles
        #  or/and also execute the algorithm with speed 1 only and merge the results to fill high speed gaps.
        if not collision and new_particle is not None and (particle_cells[new_particle.position[1], new_particle.position[0]] is None or
                particle_cells[new_particle.position[1], new_particle.position[0]].timestamp == new_particle.timestamp):
            new_particles.append(new_particle)
            if particle_cells[new_particle.position[1], new_particle.position[0]] is None:
                particle_cells[new_particle.position[1], new_particle.position[0]] = new_particle
    return new_particles


def out_of_bounds(cell_size, pos) -> bool:
    x, y = pos
    return x < 0 or x >= cell_size[1] or y < 0 or y >= cell_size[0]


def get_action_trace(position, direction, speed, action, timestamp, cells_size):
    timestamp += 1

    # update direction and speed according to action
    if action == Action.TURN_LEFT:
        direction = Direction((direction.value - 1) % 4)
    elif action == Action.TURN_RIGHT:
        direction = Direction((direction.value + 1) % 4)
    elif action == Action.SLOW_DOWN:
        speed -= 1
    elif action == Action.SPEED_UP:
        speed += 1

    # check invalid speed
    if not 1 <= speed <= 10:
        return [], None

    # empty the trace
    trace = []

    # init new pos
    new_x = position[0]
    new_y = position[1]

    # visit all cells that are within "speed"
    for i in range(speed):
        # update position
        if direction == Direction.UP:
            new_y -= 1
        elif direction == Direction.DOWN:
            new_y += 1
        elif direction == Direction.LEFT:
            new_x -= 1
        elif direction == Direction.RIGHT:
            new_x += 1
        new_pos = (new_x, new_y)
        # check borders and speed
        if out_of_bounds(cells_size, new_pos):
            return [], None

        # create trace
        # trace gaps occur every 6 rounds if the speed is higher than 2.
        if timestamp % 6 != 0 or speed < 3 or i == 1 or i == 0:
            trace.append(new_pos)

    return trace, Particle(new_pos, direction, speed, timestamp)
