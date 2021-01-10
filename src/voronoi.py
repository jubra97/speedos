from copy import copy

import numpy as np

from src.utils import Direction
import matplotlib.pyplot as plt


class Particle:

    def __init__(self, position, agent_id, direction):
        self.position = position
        self.agent_id = agent_id
        self.direction = direction


def voronoi(model, max_agent_id):
    is_endgame = True
    opponent_ids = []
    timestamp = model.schedule.steps
    cells = model.cells
    width, height = model.width, model.height
    # format: (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2), dtype=np.int16)

    particles = []
    for agent in model.active_speed_agents:
        particle = Particle(agent.pos, agent.unique_id, agent.direction)
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
                # Check for endgame here
                if particle_cells[pos[0], pos[1], 1] != 0 and bool(particle.agent_id == max_agent_id) ^ \
                        bool(particle_cells[pos[0], pos[1], 0] == max_agent_id) and \
                        particle_cells[pos[0], pos[1], 0] != -1 and particle.agent_id != -1:
                    is_endgame = False
                    if particle_cells[pos[0], pos[1], 0] not in opponent_ids:
                        opponent_ids.append(particle_cells[pos[0], pos[1], 0])
                    if particle.agent_id not in opponent_ids:
                        opponent_ids.append(particle.agent_id)
                if survived:
                    new_particles.extend(surrounding_cells(particle, width, height))

        particles = new_particles
    return particle_cells, dict(zip(*np.unique(particle_cells[:, :, 0], return_counts=True))), is_endgame, opponent_ids


def voronoi_for_reduced_opponents(model, max_agent_id, min_agent_id, is_endgame):
    timestamp = model.schedule.steps
    cells = model.cells
    width, height = model.width, model.height
    region_sizes = {max_agent_id: 0, min_agent_id: 0}
    # format: (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2), dtype=np.int16)

    particles = []
    agents_list = [model.get_agent_by_id(max_agent_id), model.get_agent_by_id(min_agent_id)] if not is_endgame else \
        [model.get_agent_by_id(max_agent_id)]
    for agent in agents_list:
        particle = Particle(agent.pos, agent.unique_id, agent.direction)
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
                    region_sizes[particle.agent_id] += 1
                elif particle_cells[pos[0], pos[1], 1] == timestamp and \
                        particle_cells[pos[0], pos[1], 0] != particle.agent_id:
                    # battlefront
                    region_sizes[particle_cells[pos[0], pos[1], 0]] -= 1  # decrease because it was falsely increased
                    particle_cells[pos[0], pos[1]] = [-1, -1]
                else:
                    survived = False
                # Check for endgame here
                if particle_cells[pos[0], pos[1], 1] != 0 and bool(particle.agent_id == max_agent_id) ^ \
                        bool(particle_cells[pos[0], pos[1], 0] == max_agent_id):
                    is_endgame = False
                if survived:
                    new_particles.extend(surrounding_cells(particle, width, height))
        particles = new_particles

    return particle_cells, region_sizes, is_endgame


def early_stop_voronoi(model, max_agent_id, min_agent_id, is_endgame, max_move, min_move):
    timestamp = model.schedule.steps
    cells = model.cells
    width, height = model.width, model.height
    nb_cells = width * height
    region_sizes = {max_agent_id: 0, min_agent_id: 0}
    # format: (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2), dtype=np.int16)

    particles = []
    agents_list = [model.get_agent_by_id(max_agent_id), model.get_agent_by_id(min_agent_id)] if not is_endgame else \
        [model.get_agent_by_id(max_agent_id)]
    for agent in agents_list:
        particle = Particle(agent.pos, agent.unique_id, agent.direction)
        particles.extend(surrounding_cells(particle, width, height))

    while len(particles) != 0:
        old_region_sizes = copy(region_sizes)
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
                    region_sizes[particle.agent_id] += 1
                elif particle_cells[pos[0], pos[1], 1] == timestamp and \
                        particle_cells[pos[0], pos[1], 0] != particle.agent_id:
                    # battlefront
                    region_sizes[particle_cells[pos[0], pos[1], 0]] -= 1    # decrease because it was falsely increased
                    particle_cells[pos[0], pos[1]] = [-1, -1]
                else:
                    survived = False
                # Check for endgame here
                if particle_cells[pos[0], pos[1], 1] != 0 and bool(particle.agent_id == max_agent_id) ^ \
                        bool(particle_cells[pos[0], pos[1], 0] == max_agent_id):
                    is_endgame = False
                if survived:
                    new_particles.extend(surrounding_cells(particle, width, height))

        particles = new_particles
        if max_move is not None and region_sizes[max_agent_id] == old_region_sizes[max_agent_id]:
            evaluation = (region_sizes[max_agent_id] - region_sizes[min_agent_id]) / nb_cells
            if evaluation < max_move:
                # max player has no more particles and is already worse than last time
                break
        elif min_move is not None and region_sizes[min_agent_id] == old_region_sizes[min_agent_id]:
            evaluation = (region_sizes[max_agent_id] - region_sizes[min_agent_id]) / nb_cells
            if evaluation > min_move:
                # min player has no more particles and is already worse than last time
                break

    return particle_cells, region_sizes, is_endgame


def surrounding_cells(parent, width, height):
    particles = []
    x, y = parent.position
    directions = [(-1, 0, Direction.LEFT), (1, 0, Direction.RIGHT), (0, -1, Direction.UP), (0, 1, Direction.DOWN)]

    # remove direction behind agent
    if parent.direction == Direction.UP:
        directions.pop(3)
    elif parent.direction == Direction.DOWN:
        directions.pop(2)
    elif parent.direction == Direction.RIGHT:
        directions.pop(0)
    elif parent.direction == Direction.LEFT:
        directions.pop(1)

    for d_x, d_y, direction in directions:
        if 0 <= x + d_x < width and 0 <= y + d_y < height:
            particles.append(Particle((x + d_x, y + d_y), parent.agent_id, direction))

    return particles
