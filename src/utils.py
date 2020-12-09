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

    particles = surrounding_cells(agent.pos, width, height)

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            if cells[particle[1], particle[0]] == 0:
                cells[particle[1], particle[0]] = marker
                new_particles.append(particle)
                reachable_cells_counter += 1
                new_particles.extend(surrounding_cells(particle, width, height))
        particles = new_particles
    return reachable_cells_counter, cells


class EasyParticle:

    def __init__(self, position, agent_id, direction):
        self.position = position
        self.agent_id = agent_id
        self.direction = direction


def speed_one_voronoi(model):
    timestamp = model.schedule.steps
    cells = model.cells
    width, height = model.width, model.height
    # (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2))

    particles = []
    for agent in model.active_speed_agents:
        particle = EasyParticle(agent.pos, agent.unique_id, agent.direction)
        particles.extend(surrounding_cells(particle, width, height))

    while len(particles) != 0:
        timestamp += 1
        new_particles = []
        for particle in particles:
            pos = (particle.position[1], particle.position[0])
            if cells[pos] == 0:
                # no obstacle in cells
                survived = True
                if particle_cells[pos[0], pos[1], 1] == 0:
                    # first
                    particle_cells[pos[0], pos[1]] = [particle.agent_id, timestamp]
                elif particle_cells[pos[0], pos[1], 1] == timestamp and \
                        particle_cells[pos[0], pos[1], 0] != particle.agent_id:
                    # battlefront
                    particle_cells[pos[0], pos[1]] = [-1, -1]
                else:
                    survived = False

                if survived:
                    new_particles.extend(surrounding_cells(particle, width, height))

        particles = new_particles
    return particle_cells


def surrounding_cells(parent, width, height):
    particles = []
    x, y = parent.position
    directions = [(-1, 0, Direction.LEFT), (1, 0, Direction.RIGHT), (0, -1, Direction.UP), (0, 1, Direction.DOWN)]

    # remove direction behind agent
    if parent.direction == Direction.UP:
        directions.remove((0, 1, Direction.DOWN))
    elif parent.direction == Direction.DOWN:
        directions.remove((0, -1, Direction.UP))
    elif parent.direction == Direction.RIGHT:
        directions.remove((-1, 0, Direction.LEFT))
    elif parent.direction == Direction.LEFT:
        directions.remove((1, 0, Direction.RIGHT))

    for d_x, d_y, direction in directions:
        if 0 <= x + d_x < width and 0 <= y + d_y < height:
            particles.append(EasyParticle((x + d_x, y + d_y), parent.agent_id, direction))

    return particles


class Particle:

    def __init__(self, position, direction, speed, timestamp, path=None, agent_id=-2, battlefront=False, trace=None):
        self.position = position
        self.direction = direction
        self.speed = speed
        self.timestamp = timestamp
        self.path = [] if path is None else path
        self.agent_id = agent_id
        self.battlefront = battlefront
        self.trace = [] if trace is None else trace


def trace_voronoi(model):
    voronoi_cells = np.zeros(shape=(*model.cells.shape, model.nb_agents))
    cells = model.cells
    particle_cells = np.empty(cells.shape, dtype=object)

    particles = []
    for agent in model.active_speed_agents:
        particles.append(Particle(agent.pos, agent.direction, agent.speed, model.schedule.steps, [], agent.unique_id))

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            agent_particles = get_new_particles(cells, particle_cells, particle, approximation=True)
            new_particles.extend(agent_particles)
            for agent_particle in agent_particles:
                if voronoi_cells[agent_particle.position[1]][agent_particle.position[0]][agent_particle.agent_id-1] == 0:
                    voronoi_cells[agent_particle.position[1]][agent_particle.position[0]][agent_particle.agent_id-1] = \
                        agent_particle.timestamp
                for pos in agent_particle.trace:
                    if voronoi_cells[pos[1]][pos[0]][agent_particle.agent_id - 1] == 0:
                        voronoi_cells[pos[1]][pos[0]][agent_particle.agent_id - 1] = agent_particle.timestamp

        particles = new_particles
    return voronoi_cells


def sync_voronoi(model, approximation=True, track_traces=False):
    voronoi_cells = None
    if track_traces:
        voronoi_cells = trace_voronoi(model)
    cells = model.cells
    particle_cells = np.empty(cells.shape, dtype=object)

    particles = []
    for agent in model.active_speed_agents:
        particles.append(Particle(agent.pos, agent.direction, agent.speed, model.schedule.steps, [], agent.unique_id))

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            agent_particles = get_new_particles(cells, particle_cells, particle, approximation, voronoi_cells)
            new_particles.extend(agent_particles)

        particles = new_particles
    return particle_cells


def voronoi(model, agent, approximation=True):
    cells = model.cells
    particle_cells = np.empty(cells.shape, dtype=object)

    init_particle = Particle(agent.pos, agent.direction, agent.speed, model.schedule.steps)
    particles = get_new_particles(cells, particle_cells, init_particle, approximation)

    while len(particles) != 0:
        new_particles = []
        for particle in particles:
            particles = get_new_particles(cells, particle_cells, particle, approximation)
            new_particles.extend(particles)

        particles = new_particles
    return particle_cells


def get_new_particles(cells, particle_cells, particle, approximation=True, voronoi_cells=None):
    new_particles = []
    for action in list(Action):
        new_particle = get_action_trace(particle, action, cells.shape)

        if not approximation and new_particle is not None:
            new_particle.path = particle.path + particle.trace

        collision = False
        for t in particle.trace:
            if voronoi_cells is not None:
                for agent_id in range(voronoi_cells.shape[2]):
                    if agent_id+1 != particle.agent_id and particle.timestamp >= voronoi_cells[t[1]][t[0]][agent_id]:
                        collision = True

            if cells[t[1], t[0]] != 0:
                collision = True
            if not approximation:
                for path_cell in particle.path:
                    if path_cell[0] == t[0] and path_cell[1] == t[1]:
                        collision = True

        if not collision and new_particle is not None:
            pos = (new_particle.position[1], new_particle.position[0])
            same_step_arrival = False
            if particle_cells[pos] is not None and particle_cells[pos].timestamp == new_particle.timestamp:
                same_step_arrival = True

            if approximation and (particle_cells[pos] is None or same_step_arrival):
                if particle_cells[pos] is not None and same_step_arrival:
                    if particle_cells[pos].agent_id == new_particle.agent_id:
                        new_particle.battlefront = particle_cells[pos].battlefront
                    else:
                        new_particle.battlefront = True
                new_particles.append(new_particle)
                particle_cells[pos] = new_particle
            elif not approximation:
                new_particles.append(new_particle)
                if particle_cells[pos] is None:
                    particle_cells[pos] = new_particle
    return new_particles


def out_of_bounds(cell_size, pos) -> bool:
    x, y = pos
    return x < 0 or x >= cell_size[1] or y < 0 or y >= cell_size[0]


def get_action_trace(particle, action, cells_size):
    timestamp = particle.timestamp + 1
    direction = particle.direction
    speed = particle.speed
    position = particle.position

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
    if not 1 <= speed <= 1:
        return None

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
            return None

        # create trace
        # trace gaps occur every 6 rounds if the speed is higher than 2.
        if timestamp % 6 != 0 or speed < 3 or i == 1 or i == 0:
            trace.append(new_pos)

    return Particle(new_pos, direction, speed, timestamp, agent_id=particle.agent_id, trace=trace)
