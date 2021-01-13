#cython: language_level=3
import time

import numpy as np

from src.core.model import SpeedModel
from src.core.utils import Direction

# cdef class Particle:
#     cdef int agent_id, direction
#     cdef int x, y
#     def __init__(self, int x, int y, int agent_id, int direction):
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id
#         self.direction = direction

cdef struct Particle:
    int x
    int y
    int agent_id
    int direction

def voronoi(model: SpeedModel, int max_agent_id):
    cdef bint is_endgame
    cdef int timestamp, width, height
    cdef (int, int) pos

    is_endgame = True
    # is_endgame = True
    opponent_ids = []
    timestamp = model.schedule.steps
    cells = model.cells
    width = model.width
    height = model.height
    # format: (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2), dtype=np.int16)

    particles = []
    cdef Particle particle
    for agent in model.active_speed_agents:
        particle = Particle(agent.pos[0], agent.pos[1], agent.unique_id, agent.direction.value)
        particles.extend(surrounding_cells(particle, width, height))

    while len(particles) != 0:
        timestamp += 1
        new_particles = []
        for particle in particles:
            pos = (particle.y, particle.x)
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

def voronoi_for_reduced_opponents(model: SpeedModel, int max_agent_id, int min_agent_id, bint is_endgame):
    cdef int timestamp, width, height
    cdef (int, int) pos
    cdef dict region_sizes
    cdef Particle particle
    cdef bint survived

    start_t = time.time()
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
        particle = Particle(agent.pos[0], agent.pos[1], agent.unique_id, agent.direction.value)
        particles.extend(surrounding_cells(particle, width, height))

    while len(particles) != 0:
        timestamp += 1
        new_particles = []
        for particle in particles:
            pos = (particle.y, particle.x)
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
    comp_time = time.time() - start_t
    return particle_cells, region_sizes, is_endgame, comp_time


def surrounding_cells(Particle parent, int width, int height):
    particles = []
    # particles = []
    cdef int x = parent.x
    cdef int y = parent.y
    cdef int d_x, d_y
    directions = [(-1, 0, Direction.LEFT), (1, 0, Direction.RIGHT), (0, -1, Direction.UP), (0, 1, Direction.DOWN)]

    # remove direction behind agent
    if parent.direction == Direction.UP.value:
        directions.pop(3)
    elif parent.direction == Direction.DOWN.value:
        directions.pop(2)
    elif parent.direction == Direction.RIGHT.value:
        directions.pop(0)
    elif parent.direction == Direction.LEFT.value:
        directions.pop(1)

    for d_x, d_y, direction in directions:
        if 0 <= x + d_x < width and 0 <= y + d_y < height:
            particles.append(Particle(x + d_x, y + d_y, parent.agent_id, direction.value))
    return particles
