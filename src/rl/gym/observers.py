from gym import spaces
import numpy as np
from src.utils import Direction
from PIL import Image
import cv2


class GlobalObserver:

    def __init__(self, nb_agents, width, height):
        self.nb_agents = nb_agents
        self.width = width
        self.height = height
        lows = np.zeros(width * height + nb_agents * 5 + 1)
        highs = np.ones(width * height + nb_agents * 5 + 1)
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

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


class GlobalMixedImageObserver:

    def __init__(self, nb_agents, width, height):
        self.nb_agents = nb_agents
        self.width = width
        self.height = height
        lows = np.zeros((height, width, 2))
        highs = np.full((height, width, 2), 255)
        self.observation_space = spaces.Box(low=lows, high=highs, shape=(height, width, 2), dtype='uint8')

    def prepared_state(self, state, agent_id, step_counter):
        # format: [R, G, B] image with R=Cells, G=Empty, B=Scalar_Information_with_Padding
        # every input is normalized to [0, 255]
        original_cells = state["cells"]
        width = state["width"]
        height = state["height"]

        # set every cell to 0 (empty) or 1 (occupied)
        # it is important to create a new cells array in order to not overwrite the original one
        cells = np.zeros((height, width, 2), dtype='uint8')
        for x in range(width):
            for y in range(height):
                if original_cells[y][x] == 0:
                    cells[y][x][0] = 0
                else:
                    cells[y][x][0] = 255
        pixels = cv2.resize(cells, dsize=(self.height, self.width), interpolation=cv2.INTER_NEAREST)

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
            # one hot for direction
            direction = np.zeros(4)
            direction[Direction[player["direction"].upper()].value] = 1
            player_inf.extend(direction)
            player_inf.append(player["speed"] / 10)
            player_inf.append(1 if player["active"] else 0)

        img_id_counter = 0
        for player_attr in player_inf:
            r, g = pixels[0, img_id_counter]
            pixels[0, img_id_counter] = (r, int(player_attr * 255))
            img_id_counter += 1

        # 6 round counter
        r, g = pixels[0, img_id_counter]
        pixels[0, img_id_counter] = (r, int((step_counter % 6) / 5 * 255))

        #img = Image.fromarray(pixels, 'RGB')
        #img.show()

        return pixels


class GlobalImageObserver:

    def __init__(self, nb_agents, width, height):
        self.nb_agents = nb_agents
        self.width = width
        self.height = height
        lows = np.zeros((height, width, 3))
        highs = np.full((height, width, 3), 255)
        self.observation_space = spaces.Box(low=lows, high=highs, shape=(height, width, 3), dtype='uint8')

    def prepared_state(self, state, agent_id, step_counter):
        # format: [R, G, B, A] image with R=Obstacles, G=Opponents, B=Self, A=Round_Counter
        # every input is normalized to [0, 255]
        original_cells = state["cells"]
        width = state["width"]
        height = state["height"]

        cells = np.zeros((height, width, 3), dtype='uint8')
        for x in range(width):
            for y in range(height):
                # set every cell to 0 (empty) or 1 (occupied)
                # it is important to create a new cells array in order to not overwrite the original one
                if original_cells[y, x] == 0:
                    cells[y, x, 0] = 0
                else:
                    cells[y, x, 0] = 255

                # put the 6 round counter on the 4th layer
                #cells[y, x, 3] = 255 #int((step_counter % 6) / 5 * 255)

        # player information is put into layers 2 and 3 (g, b)
        other_player_ids = list(state["players"].keys())
        other_player_ids.remove(str(agent_id))
        for player_id in other_player_ids:
            player = state["players"][player_id]
            if player["active"]:
                self.add_player_to_layer(cells, 1, player)
        self.add_player_to_layer(cells, 2, state["players"][str(agent_id)])

        pixels = cv2.resize(cells, dsize=(self.height, self.width), interpolation=cv2.INTER_NEAREST)

        # img = Image.fromarray(pixels, 'RGB')
        # img.show()
        return pixels

    @staticmethod
    def add_player_to_layer(cells, layer_id, player):
        x, y = player["x"], player["y"]
        if 0 < player["y"] >= cells.shape[0] or 0 < player["x"] >= cells.shape[1]:
            return
        # 127 to mark the player + speed indication
        cells[y, x, layer_id] += 127 + player["speed"] / 10 * 64

        direction = Direction[player["direction"].upper()]
        if direction == Direction.UP:
            x -= 1
        elif direction == Direction.DOWN:
            x += 1
        elif direction == Direction.LEFT:
            y -= 1
        elif direction == Direction.RIGHT:
            y += 1
        x = max(min(x, cells.shape[1] - 1), 0)
        y = max(min(y, cells.shape[0] - 1), 0)
        # indicate direction with 16 (16, so that the value on a pixel can at most add up to 255)
        cells[y, x, layer_id] += 64
