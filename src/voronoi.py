import numpy as np

from src.utils import Direction


class Particle:

    def __init__(self, position, agent_id, direction):
        self.position = position
        self.agent_id = agent_id
        self.direction = direction


def voronoi(model, max_agent_id):
    is_endgame = True
    timestamp = model.schedule.steps
    cells = model.cells
    width, height = model.width, model.height
    # format: (height, width, (id, timestamp))
    particle_cells = np.zeros((*cells.shape, 2), dtype=np.int8)

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
                        bool(particle_cells[pos[0], pos[1], 0] == max_agent_id):
                    is_endgame = False
                if survived:
                    new_particles.extend(surrounding_cells(particle, width, height))

        particles = new_particles
    return particle_cells, dict(zip(*np.unique(particle_cells[:, :, 0], return_counts=True))), is_endgame


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
