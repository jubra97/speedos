import datetime
import multiprocessing
from itertools import permutations

import numpy as np
import requests
from pynput import keyboard

from src.model import SpeedAgent
from src.utils import Action, get_state, arg_maxes, state_to_model, model_to_json
from src.voronoi import voronoi
import numpy as np
import copy
from scipy.spatial import distance


class AgentDummy(SpeedAgent):
    """
    Agent dummy for reinforcement learning.
    It doesn't choose and set an action since the Gym environment controls the execution.
    """

    def act(self, state):
        return None

    def step(self):
        pass


class RandomAgent(SpeedAgent):
    """
    Agent that chooses random actions.
    """

    def act(self, state):
        own_id = state["you"]
        own_props = state["players"][str(own_id)]
        possible_actions = list(Action)
        if own_props["speed"] == 1:
            possible_actions.remove(Action.SLOW_DOWN)
        elif own_props["speed"] == 10:
            possible_actions.remove(Action.SPEED_UP)
        return self.random.choice(possible_actions)


class NStepSurvivalAgent(SpeedAgent):
    """
    Agent that calculates all action combinations for the next n (depth) steps and chooses the action that has the
    lowest amount of death paths.
    """
    def __init__(self, model, pos, direction, speed=1, active=True, depth=1, deterministic=False):
        super().__init__(model, pos, direction, speed, active)
        self.depth = depth
        self.survival = None
        self.deterministic = deterministic

    def act(self, state):
        self.survival = dict.fromkeys(list(Action), 0)
        self.deep_search(state, self.depth, None)
        amaxes = arg_maxes(self.survival.values(), list(self.survival.keys()))
        if len(amaxes) == 0:
            amaxes = list(Action)
        if self.deterministic:
            return amaxes[0]
        else:
            return np.random.choice(amaxes)

    def deep_search(self, state, depth, initial_action):
        own_id = state["you"]

        if not state["players"][str(own_id)]["active"]:
            return
        elif depth == 0:
            self.survival[initial_action] += 1
        else:
            model = state_to_model(state)
            nb_active_agents = len(model.active_speed_agents)
            action_permutations = list(permutations(list(Action), nb_active_agents))
            for action_permutation in action_permutations:
                own_agent = model.get_agent_by_id(own_id)
                for idx, agent in enumerate(model.active_speed_agents):
                    agent.action = action_permutation[idx]
                model.step()
                new_state = get_state(model, own_agent, self.deadline)
                # recursion
                if initial_action is None:
                    self.deep_search(new_state, depth - 1, own_agent.action)
                else:
                    self.deep_search(new_state, depth - 1, initial_action)
                model = state_to_model(state)


class HumanAgent(SpeedAgent):

    def act(self, state):
        with keyboard.Events() as events:
            # Block for as much as possible
            input_key = events.get(1000000).key

        if input_key == keyboard.KeyCode.from_char('w'):
            return Action.SPEED_UP
        elif input_key == keyboard.KeyCode.from_char('s'):
            return Action.SLOW_DOWN
        elif input_key == keyboard.KeyCode.from_char('a'):
            return Action.TURN_LEFT
        elif input_key == keyboard.KeyCode.from_char('d'):
            return Action.TURN_RIGHT
        else:
            return Action.CHANGE_NOTHING


class BaseMultiMiniMaxAgent(SpeedAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm
    """
    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=1):
        super().__init__(model, pos, direction, speed, active)
        self.time_for_move = time_for_move
        self.max_cache_depth = 4
        self.depth = 2

    def act(self, state):
        globals()["cache"] = dict()  # defaultdict(int)

        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=self.depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state))
        p.start()
        p.join(self.time_for_move)

        # If thread is active
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()

        print(f"reached_depth: {reached_depth.value}")

        return Action(move.value)

    def depth_first_iterative_deepening(self, shared_move_var, shared_dep, game_state):
        while True:  # canceled from outside
            move_to_make = self.multi_minimax(self.depth, game_state)
            shared_move_var.value = move_to_make.value
            shared_dep.value = self.depth
            self.depth += 1

    def multi_minimax(self, depth, game_state):
        model = state_to_model(game_state)
        own_id = game_state["you"]
        _, _, is_endgame = voronoi(model, own_id)
        max_player = model.get_agent_by_id(own_id)
        min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
        min_player_ids.remove(own_id)

        move_to_make = Action.CHANGE_NOTHING
        max_move = float("-inf")
        alpha = float("-inf")
        copy_actions = copy.deepcopy(list(Action))
        for action in reversed(copy_actions):
            if max_player.speed == 1 and action == Action.SLOW_DOWN:
                continue
            if max_player.speed == 10 and action == Action.SPEED_UP:
                continue
            pre_state = model_to_json(model, trace_aware=True)
            max_player.action = action
            model.step_specific_agent(max_player)
            tree_path = str(action.value)
            min_move = float("inf")
            beta = float("inf")
            for opponent_id in min_player_ids:
                opponent = model.get_agent_by_id(opponent_id)

                min_move = min(min_move, self.minimax(max_player, opponent, depth - 1, alpha, beta, False,
                                                      model, is_endgame, tree_path=tree_path))

                model = state_to_model(pre_state, trace_aware=True)
                max_player = model.get_agent_by_id(own_id)
                beta = min_move

                if alpha >= beta:
                    break
                if is_endgame:
                    break

            if min_move >= max_move:
                if min_move == max_move:
                    move_to_make = np.random.choice([move_to_make, action])
                else:
                    move_to_make = action
                max_move = min_move
                alpha = max_move
        return move_to_make

    def minimax(self, max_player, min_player, depth, alpha, beta, is_max, model, is_endgame, tree_path=None):
        if depth == 0 or not max_player.active or not model.running:
            return self.evaluate_position(model, max_player, min_player, depth, tree_path=tree_path)
        if is_max or is_endgame:
            max_move = float("-inf")
            pre_state = model_to_json(model, trace_aware=True)
            sorted_action_list = [Action.CHANGE_NOTHING, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPEED_UP, Action.SLOW_DOWN]
            for action in sorted_action_list:
                max_player.action = action
                model.step_specific_agent(max_player)
                tree_path += str(action.value)

                max_move = max(max_move, self.minimax(max_player, min_player, depth - 1, alpha, beta, False,
                                                      model, is_endgame, tree_path=tree_path))

                model = state_to_model(pre_state, trace_aware=True)
                own_id = max_player.unique_id
                max_player = model.get_agent_by_id(own_id)
                min_player_id = min_player.unique_id
                min_player = model.get_agent_by_id(min_player_id)

                alpha = max(alpha, max_move)
                if alpha >= beta:
                    break
            return max_move
        else:
            min_move = float("inf")
            pre_state = model_to_json(model, trace_aware=True)
            sorted_action_list = [Action.CHANGE_NOTHING, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPEED_UP, Action.SLOW_DOWN]
            for action in list(sorted_action_list):
                min_player.action = action
                model.step_specific_agent(min_player)
                tree_path += str(action.value)

                min_move = min(min_move, self.minimax(max_player, min_player, depth - 1, alpha, beta, True,
                                                      model, is_endgame, tree_path=tree_path))

                model = state_to_model(pre_state, trace_aware=True)
                own_id = max_player.unique_id
                max_player = model.get_agent_by_id(own_id)
                min_player_id = min_player.unique_id
                min_player = model.get_agent_by_id(min_player_id)

                beta = min(beta, min_move)
                if alpha >= beta:
                    break
            return min_move

    def evaluate_position(self, model, max_player, min_player, depth, tree_path=None,
                          caching_enabled=False):
        # use cached value
        # print("evaluate " + tree_path)
        if caching_enabled and globals()["cache"] is not None:
            cache_key = tree_path  # hash_state(state)
            if cache_key in globals()["cache"]:
                print("found cached " + tree_path)
                return globals()["cache"][cache_key]
        max_depth = self.depth

        if max_player.active and not min_player.active:
            return float("inf")
        else:
            # subtract a high number if agent died (otherwise dying in the last step is as good as survival)
            if not max_player.active:
                death_penalty = 1000
            else:
                death_penalty = 0
            return -1 * depth - death_penalty


class VoronoiMultiMiniMaxAgent(BaseMultiMiniMaxAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation
    """
    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=5):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.time_for_move = time_for_move
        self.max_cache_depth = 4
        self.depth = 2

    def evaluate_position(self, model, max_player, min_player, depth, tree_path=None, caching_enabled=False):
        if caching_enabled and globals()["cache"] is not None:
            cache_key = tree_path  # hash_state(state)
            if cache_key in globals()["cache"]:
                print("found cached " + tree_path)
                return globals()["cache"][cache_key]

        # weights - all non-weighted evaluation values should be in [-1, 1]
        # using very large weight gaps is effectively like prioritization
        kill_weight = float("inf")
        death_weight = 1000
        voronoi_region_weight = 1
        territory_bonus_weight = 0.001  # only used to decide between positions with equal voronoi region evaluation

        # TODO: Bonus points for adjacent battlezones

        if max_player.active and not min_player.active:
            utility = kill_weight
            if caching_enabled and depth < self.max_cache_depth and globals()["cache"] is not None:
                globals()["cache"][cache_key] = utility  # cache result
            return utility
        elif not max_player.active:
            # TODO: Detect situations where the game is lost if we try to survive but we can force a kamikaze draw.
            #       At the moment the agent would always try to survive one more step and would only kamikaze if death
            #       is inevitable.
            death_eval = -depth - 1  # -1 to keep negative when depth is 0
            if not min_player.active:
                # kamikaze is better than just dying without killing someone else
                death_eval += 0.1
            utility = death_eval / self.depth * death_weight
            if caching_enabled and depth < self.max_cache_depth and globals()["cache"] is not None:
                globals()["cache"][cache_key] = utility  # cache result
            return utility

        voronoi_cells, voronoi_counter, is_endgame = voronoi(model, max_player.unique_id)
        nb_cells = float(model.width * model.height)  # for normalization

        # voronoi region size comparison
        max_player_size = voronoi_counter[
            max_player.unique_id] if max_player.unique_id in voronoi_counter.keys() else 0
        min_player_size = voronoi_counter[
            min_player.unique_id] if min_player.unique_id in voronoi_counter.keys() else 0
        voronoi_region_points = (max_player_size - min_player_size) / nb_cells * voronoi_region_weight

        territory_bonus = 0
        for x in range(model.width):
            for y in range(model.height):
                if voronoi_cells[y, x, 0] == max_player.unique_id:
                    territory_bonus += add_territory_bonus(model, x, y)
        territory_bonus *= territory_bonus_weight / nb_cells

        utility = voronoi_region_points + territory_bonus
        if caching_enabled and depth < self.max_cache_depth and globals()["cache"] is not None:
            globals()["cache"][cache_key] = utility  # cache result
        return utility


def add_territory_bonus(model, x, y):
    territory_bonus = 0.25
    bonus = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d_x, d_y in directions:
        if 0 <= x + d_x < model.width and 0 <= y + d_y < model.height and model.cells[y + d_y, x + d_x] == 0:
            bonus += territory_bonus

    return bonus


class ReduceOpponentsVoronoiMultiMiniMaxAgent(VoronoiMultiMiniMaxAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=5):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.time_for_move = time_for_move
        self.max_cache_depth = 4
        self.depth = 2

    def multi_minimax(self, depth, game_state):
        model = state_to_model(game_state)
        own_id = game_state["you"]
        _, _, is_endgame = voronoi(model, own_id)
        max_player = model.get_agent_by_id(own_id)
        min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
        min_player_ids.remove(own_id)
        min_player_ids.sort(key=lambda id: distance.euclidean(model.get_agent_by_id(id).pos, max_player.pos), reverse=True)
        if len(min_player_ids) >= 2:
            min_player_ids = min_player_ids[-2:]

        move_to_make = Action.CHANGE_NOTHING
        max_move = float("-inf")
        alpha = float("-inf")
        sorted_action_list = [Action.CHANGE_NOTHING, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPEED_UP, Action.SLOW_DOWN]
        for action in sorted_action_list:
            if max_player.speed == 1 and action == Action.SLOW_DOWN:
                continue
            if max_player.speed == 10 and action == Action.SPEED_UP:
                continue
            pre_state = model_to_json(model, trace_aware=True)
            max_player.action = action
            model.step_specific_agent(max_player)
            tree_path = str(action.value)
            min_move = float("inf")
            beta = float("inf")
            for opponent_id in min_player_ids:
                opponent = model.get_agent_by_id(opponent_id)

                min_move = min(min_move, self.minimax(max_player, opponent, depth - 1, alpha, beta, False,
                                                      model, is_endgame, tree_path=tree_path))

                model = state_to_model(pre_state, trace_aware=True)
                max_player = model.get_agent_by_id(own_id)
                beta = min_move

                if alpha >= beta:
                    break
                if is_endgame:
                    break

            if min_move >= max_move:
                if min_move == max_move:
                    move_to_make = np.random.choice([move_to_make, action])
                else:
                    move_to_make = action
                max_move = min_move
                alpha = max_move
        return move_to_make


class LiveAgent(ReduceOpponentsVoronoiMultiMiniMaxAgent):
    """
    Live Agent
    """
    def __init__(self, model, pos, direction, speed=1, active=True):
        super().__init__(model, pos, direction, speed, active)
        self.max_cache_depth = 4
        self.depth = 2

    def act(self, state):
        globals()["cache"] = dict()  # defaultdict(int)

        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=self.depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state))
        p.start()
        send_time = 1
        deadline = datetime.datetime.strptime(state["deadline"], "%Y-%m-%dT%H:%M:%SZ")
        response = requests.get("https://msoll.de/spe_ed_time")
        server_time = datetime.datetime.strptime(response.json()["time"], "%Y-%m-%dT%H:%M:%SZ")
        av_time = (deadline - server_time).total_seconds() - send_time
        p.join(av_time)

        # If thread is active
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()

        print(f"reached_depth: {reached_depth.value}")
        return Action(move.value)