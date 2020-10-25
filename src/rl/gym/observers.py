from gym import spaces
import numpy as np
from src.utils import Direction


class GlobalObserver:

    def __init__(self, nb_agents, width, height):
        self.nb_agents = nb_agents
        self.width = width
        self.height = height
        lows = np.zeros(width * height + nb_agents * 5 + 1)
        highs = np.ones(width * height + nb_agents * 5 + 1)
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.int)

    @staticmethod
    def prepared_state(state, agent_id, step_counter):
        # format: [cells, players_information, 6_round_counter]
        # every input is normalized to [0, 1]
        original_cells = state["cells"]
        width = state["width"]
        height = state["height"]

        # set every cell to 0 (empty) or 1 (occupied)
        # it is important to create a new cells array in order to not overwrite the original one
        cells = np.empty(original_cells.shape)
        for x in range(width):
            for y in range(height):
                if original_cells[y][x] == 0:
                    cells[y][x] = 0
                else:
                    cells[y][x] = 1

        # player information for every player in form of [x, y, direction, speed, active]
        player_inf = []
        # put own agent information to the front so that is indirectly given which one it is
        player_ids = list(state["players"].keys())
        player_ids.remove(str(agent_id))
        player_ids = [str(agent_id)] + player_ids
        for player_id in player_ids:
            player = state["players"][player_id]
            player_inf.append(player["x"] / height)
            player_inf.append(player["y"] / width)
            player_inf.append(Direction[player["direction"].upper()].value / 4)
            player_inf.append(player["speed"] / 10)
            player_inf.append(1 if player["active"] else 0)

        # 6 round counter
        step_6_counter = (step_counter % 6) / 5

        return np.asarray([
            *np.asarray(cells).flatten(),
            *player_inf,
            step_6_counter
        ])
