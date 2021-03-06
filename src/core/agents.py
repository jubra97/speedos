import copy
import datetime
import multiprocessing
import time
from itertools import permutations

import numpy as np
import requests
from scipy.spatial import distance

from src.core.model import SpeedAgent
from src.core.utils import Action, get_state, arg_maxes, state_to_model, model_to_json, reduce_state_to_sliding_window
from src.core.voronoi_cython import voronoi, voronoi_for_reduced_opponents


class DummyAgent(SpeedAgent):
    """
    Agent with no action implementation.
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
    """
    Agent that allows to play our simulation interactively.
    """
    def act(self, state):
        from pynput import keyboard
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


class MultiMinimaxAgent(SpeedAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm.
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=4):
        super().__init__(model, pos, direction, speed, active)
        self.time_for_move = time_for_move
        self.depth = 2
        self.game_step = 0

    def step(self):
        super().step()
        self.game_step += 1

    def act(self, state):
        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=self.depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state))
        p.start()
        p.join(self.time_for_move)

        # Force termination
        if p.is_alive():
            p.terminate()
            p.join()

        return Action(move.value)

    def depth_first_iterative_deepening(self, shared_move_var, shared_dep, game_state):
        while True:  # canceled from outside
            move_to_make = self.multi_minimax(self.depth, game_state)
            shared_move_var.value = move_to_make.value
            shared_dep.value = self.depth
            self.depth += 1


    def multi_minimax(self, depth, game_state):
        model, max_player, min_player_ids, is_endgame, move_to_make, max_move, alpha, actions = \
            self.init_multi_minimax(game_state)
        for action in actions:
            if action == Action.SLOW_DOWN and max_player.speed == 1:
                continue
            if action == Action.SPEED_UP and max_player.speed >= 3 and not is_endgame:
                continue
            pre_state = model_to_json(model, trace_aware=True, step=True)
            self.update_model(model, max_player, action)
            if is_endgame:
                model.schedule.steps += 1
            min_move = float("inf")

            model, max_player, min_move, alpha = self.move_min_players(model, max_player, min_player_ids, min_move,
                                                                       depth, alpha, is_endgame, pre_state)
            move_to_make, max_move, alpha = self.update_move_to_make(min_move, move_to_make, action, max_move, alpha)

        return move_to_make

    def init_multi_minimax(self, game_state):
        game_state["step"] = self.game_step
        model = state_to_model(game_state)
        own_id = game_state["you"]
        _, _, is_endgame, _ = voronoi(model, own_id)
        max_player = model.get_agent_by_id(own_id)

        min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
        min_player_ids.remove(own_id)

        move_to_make = Action.CHANGE_NOTHING
        max_move = float("-inf")
        alpha = float("-inf")
        actions = self.init_actions()
        return model, max_player, min_player_ids, is_endgame, move_to_make, max_move, alpha, actions

    def move_min_players(self, model, max_player, min_player_ids, min_move, depth, alpha, is_endgame, pre_state):
        beta = float("inf")
        for min_player_id in min_player_ids:
            min_player = model.get_agent_by_id(min_player_id)
            min_move = min(min_move, self.minimax(max_player, min_player, depth - 1, alpha, beta, False, model,
                                                  is_endgame))
            model, max_player, _ = self.reset_model(pre_state, max_player, min_player)

            beta = min_move
            if alpha >= beta or is_endgame:
                break
        return model, max_player, min_move, alpha

    @staticmethod
    def update_move_to_make(min_move, move_to_make, action, max_move, alpha):
        if min_move >= max_move:
            if min_move == max_move:
                move_to_make = np.random.choice([move_to_make, action])
            else:
                move_to_make = action
            max_move = min_move
            alpha = max_move
        return move_to_make, max_move, alpha

    def minimax(self, max_player, min_player, depth, alpha, beta, is_max, model, is_endgame):
        # termination
        if depth == 0 or not max_player.active or not model.running:
            return self.evaluate_position(model, max_player, min_player, depth)

        # recursion
        if is_max or is_endgame:
            return self.max(max_player, min_player, depth, alpha, beta, model, is_endgame)
        else:
            return self.min(max_player, min_player, depth, alpha, beta, model, is_endgame)

    def max(self, max_player, min_player, depth, alpha, beta, model, is_endgame):
        max_move = float("-inf")
        pre_state = model_to_json(model, trace_aware=True, step=True)
        sorted_action_list = self.init_actions()

        for action in sorted_action_list:
            if action == Action.SLOW_DOWN and max_player.speed == 1:
                continue
            if action == Action.SPEED_UP and max_player.speed >= 3 and not is_endgame:
                continue

            self.update_model(model, max_player, action)
            if is_endgame:
                model.schedule.steps += 1
            max_move = max(max_move, self.minimax(max_player, min_player, depth - 1, alpha, beta, False, model,
                                                  is_endgame))
            model, max_player, min_player = self.reset_model(pre_state, max_player, min_player)

            alpha = max(alpha, max_move)
            if alpha >= beta:
                break
        return max_move

    def min(self, max_player, min_player, depth, alpha, beta, model, is_endgame):
        min_move = float("inf")
        pre_state = model_to_json(model, trace_aware=True, step=True)
        sorted_action_list = self.init_actions()

        for action in list(sorted_action_list):

            self.update_model(model, min_player, action)
            model.schedule.steps += 1
            min_move = min(min_move, self.minimax(max_player, min_player, depth - 1, alpha, beta, True, model,
                                                  is_endgame))
            model, max_player, min_player = self.reset_model(pre_state, max_player, min_player)

            beta = min(beta, min_move)
            if alpha >= beta:
                break
        return min_move

    @staticmethod
    def init_actions():
        return [Action.CHANGE_NOTHING, Action.TURN_LEFT, Action.TURN_RIGHT, Action.SPEED_UP, Action.SLOW_DOWN]

    @staticmethod
    def update_model(model, player, action):
        player.action = action
        model.step_specific_agent(player)

    @staticmethod
    def reset_model(pre_state, max_player, min_player):
        model = state_to_model(pre_state, trace_aware=True)
        own_id = max_player.unique_id
        max_player = model.get_agent_by_id(own_id)
        min_player_id = min_player.unique_id
        min_player = model.get_agent_by_id(min_player_id)
        return model, max_player, min_player

    def evaluate_position(self, model, max_player, min_player, depth):
        sub_evaluations = [self.win_evaluation, self.death_evaluation]
        weights = [float("inf"), 1]
        result = self.prioritised_evaluation(sub_evaluations, weights, model, max_player, min_player, depth)

        return result

    @staticmethod
    def win_evaluation(model, max_player, min_player, depth):
        if max_player.active and not min_player.active:
            return 1
        else:
            return None

    @staticmethod
    def death_evaluation(model, max_player, min_player, depth):
        if not max_player.active:
            # subtract a high number if agent died (otherwise dying in the last step is as good as survival)
            death_penalty = 1000
        else:
            death_penalty = 0
        return -depth - death_penalty

    @staticmethod
    def prioritised_evaluation(sub_evaluations, weights, *args):
        """
        Executes sub evaluations prioritized by the given order and multiplies results with the given weights.
        Each sub evaluation function has to return a evaluation value in range [-1, 1] or None. If the evaluation
        function returns None, the sub evaluation with the next highest priority will be executed. The evaluation
        function with the lowest priority has to return an evaluation value (not None).
        :param sub_evaluations: A list of evaluation functions with the first having the highest priority.
        :param weights: A list of weights that are multiplied with the respective evaluation values from sub
                        evaluations.
        :param args: Arguments that are passed to all sub evaluation functions
        :return: The overall evaluation value
        """
        result = None
        for i, sub_evaluation in enumerate(sub_evaluations):
            result = sub_evaluation(*args)
            if result is not None:
                return weights[i] * result
        return result


class VoronoiAgent(MultiMinimaxAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=5):
        super().__init__(model, pos, direction, speed, active, time_for_move)

    def evaluate_position(self, model, max_player, min_player, depth):
        sub_evaluations = [self.win_evaluation, self.death_evaluation, self.voronoi_evaluation]
        weights = [float("inf"), 1000, 1]
        result = self.prioritised_evaluation(sub_evaluations, weights, model, max_player, min_player, depth)

        return result

    def death_evaluation(self, model, max_player, min_player, depth):
        if not max_player.active:
            death_eval = -depth - 1  # -1 to keep negative when depth is 0
            if not min_player.active:
                # kamikaze is better than just dying without killing someone else
                death_eval += 0.1
            return death_eval / self.depth

    def voronoi_evaluation(self, model, max_player, min_player, depth):
        # only use the territory bonus as a tie breaker between positions with equal voronoi value
        territory_bonus_weight = 0.0001
        # for normalization
        nb_cells = float(model.width * model.height)

        # calculate and compare voronoi region sizes
        voronoi_cells, max_player_size, min_player_size = self.voronoi_region_sizes(model, max_player, min_player)
        voronoi_region_points = (max_player_size - min_player_size) / nb_cells

        # calculate territory bonus
        territory_bonus = self.territory_bonus(model, voronoi_cells, max_player)
        territory_bonus *= territory_bonus_weight / nb_cells

        return voronoi_region_points + territory_bonus

    @staticmethod
    def voronoi_region_sizes(model, max_player, min_player):
        voronoi_cells, voronoi_counter, _, _ = voronoi(model, max_player.unique_id)
        max_player_size = voronoi_counter[
            max_player.unique_id] if max_player.unique_id in voronoi_counter.keys() else 0
        min_player_size = voronoi_counter[
            min_player.unique_id] if min_player.unique_id in voronoi_counter.keys() else 0
        return voronoi_cells, max_player_size, min_player_size

    def territory_bonus(self, model, voronoi_cells, max_player):
        territory_bonus = 0
        for x in range(model.width):
            for y in range(model.height):
                if voronoi_cells[y, x, 0] == max_player.unique_id:
                    territory_bonus += self.cell_territory_bonus(model, x, y)
        return territory_bonus

    @staticmethod
    def cell_territory_bonus(model, x, y):
        territory_bonus = 0.25
        bonus = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d_x, d_y in directions:
            if 0 <= x + d_x < model.width and 0 <= y + d_y < model.height and model.cells[y + d_y, x + d_x] == 0:
                bonus += territory_bonus

        return bonus


class ParallelVoronoiAgent(VoronoiAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation. It is extended
    by multiprocessing to speed up the computation time.
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=5):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.start_depth = 2
        self.reached_depth = (False, 0)
        self.move_to_make = 4
        self.sub_evaluations = [self.win_evaluation, self.death_evaluation, self.voronoi_evaluation]
        self.weights = [float("inf"), 1000, 1]
        self.cores = multiprocessing.cpu_count() - 1
        if self.cores > 6:
            self.cores = 6

    def act(self, state):
        self.reached_depth = (False, 0)
        self.depth_first_iterative_deepening(state)
        return Action(self.move_to_make)

    def depth_first_iterative_deepening(self, game_state):
        def compare_depth(result):
            if result["depth"] > self.reached_depth[1] or (not self.reached_depth[0] and result["with_voronoi"]):
                self.reached_depth = (result["with_voronoi"], result["depth"])
                self.move_to_make = result["move_to_make"]

        p = multiprocessing.Pool(self.cores)

        # also compute minimax without voronoi for depth 1 to not crash in others if voronoi computation needs too long.
        self.sub_evaluations = [self.win_evaluation, MultiMinimaxAgent.death_evaluation]
        self.weights = [float("inf"), 1]
        [p.apply_async(self.depth_first_iterative_deepening_one_depth, (copy.deepcopy(game_state), depth, False),
                       callback=compare_depth) for depth in [2]]
        time.sleep(0.01)  # without sleep self.sub_evaluations is changed too early.
        self.sub_evaluations = [self.win_evaluation, self.death_evaluation, self.voronoi_evaluation]
        self.weights = [float("inf"), 1000, 1]

        [p.apply_async(self.depth_first_iterative_deepening_one_depth, (copy.deepcopy(game_state), depth, True),
                       callback=compare_depth) for depth in range(self.start_depth, 100)]
        time.sleep(self.time_for_move)
        p.terminate()

    def depth_first_iterative_deepening_one_depth(self, game_state, depth, with_voronoi):
        self.depth = depth
        move_to_make = self.multi_minimax(depth, game_state)
        return {"depth": depth, "move_to_make": move_to_make.value, "with_voronoi": with_voronoi}

    def evaluate_position(self, model, max_player, min_player, depth):
        result = self.prioritised_evaluation(self.sub_evaluations, self.weights, model, max_player, min_player, depth)
        return result


class ClosestOpponentsVoronoiAgent(VoronoiAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation. Only reachable
    opponents are considered.
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=4):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.is_endgame = False

    def init_multi_minimax(self, game_state):
        game_state["step"] = self.game_step
        model = state_to_model(game_state)
        own_id = game_state["you"]
        _, _, is_endgame, min_player_ids = voronoi(model, own_id)
        self.is_endgame = is_endgame
        max_player = model.get_agent_by_id(own_id)

        if own_id in min_player_ids:
            min_player_ids.remove(own_id)

        if len(min_player_ids) == 0:
            min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
            min_player_ids.remove(own_id)

        move_to_make = Action.CHANGE_NOTHING
        max_move = float("-inf")
        alpha = float("-inf")
        actions = self.init_actions()
        return model, max_player, min_player_ids, is_endgame, move_to_make, max_move, alpha, actions

    def voronoi_region_sizes(self, model, max_player, min_player):
        voronoi_cells, voronoi_counter, _ = voronoi_for_reduced_opponents(model, max_player.unique_id,
                                                                          min_player.unique_id,
                                                                          self.is_endgame)
        max_player_size = voronoi_counter[
            max_player.unique_id] if max_player.unique_id in voronoi_counter.keys() else 0
        min_player_size = voronoi_counter[
            min_player.unique_id] if min_player.unique_id in voronoi_counter.keys() else 0
        return voronoi_cells, max_player_size, min_player_size


class SlidingWindowVoronoiAgent(ClosestOpponentsVoronoiAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation. The model for
    the simulation is reduced by computing a sliding window that moves with the head of our agent.
    """

    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=4, min_sliding_window_size=20,
                 sliding_window_size_offset=5):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.min_sliding_window_size = min_sliding_window_size
        self.sliding_window_size_offset = sliding_window_size_offset

    def act(self, state):
        model = state_to_model(state)
        own_id = state["you"]
        _, _, is_endgame, min_player_ids = voronoi(model, own_id)
        if own_id in min_player_ids:
            min_player_ids.remove(own_id)
        if not is_endgame and len(min_player_ids) > 1:
            pos = model.get_agent_by_id(own_id).pos
            opponent_pos = model.get_agent_by_id(min_player_ids[0]).pos
            distance_to_next_opponent = distance.euclidean(pos, opponent_pos)
            state = reduce_state_to_sliding_window(state,
                                                   distance_to_next_opponent,
                                                   min_sliding_window_size=self.min_sliding_window_size,
                                                   sliding_window_size_offset=self.sliding_window_size_offset)

        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=self.depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state))
        p.start()
        p.join(self.time_for_move)

        # Force termination
        if p.is_alive():
            p.terminate()
            p.join()

        return Action(move.value)


class ParallelSlidingWindowVoronoiAgent(ParallelVoronoiAgent):
    """
    Agent that chooses an action based on the multi minimax algorithm and uses voronoi as evaluation. The model for
    the simulation is reduced by computing a sliding window that moves with the head of our agent. It is extended
    by multiprocessing to speed up the computation time.
    """
    def __init__(self, model, pos, direction, speed=1, active=True, time_for_move=4, min_sliding_window_size=12,
                 sliding_window_size_offset=3):
        super().__init__(model, pos, direction, speed, active, time_for_move)
        self.min_sliding_window_size = min_sliding_window_size
        self.sliding_window_size_offset = sliding_window_size_offset
        self.is_endgame = False

    def act(self, state):
        self.reached_depth = (False, 0)
        model = state_to_model(state)
        own_id = state["you"]
        _, _, is_endgame, min_player_ids = voronoi(model, own_id)
        if own_id in min_player_ids:
            min_player_ids.remove(own_id)
        self.is_endgame = is_endgame
        if not is_endgame and len(min_player_ids) > 1:
            pos = model.get_agent_by_id(own_id).pos
            opponent_pos = model.get_agent_by_id(min_player_ids[0]).pos
            distance_to_next_opponent = distance.euclidean(pos, opponent_pos)
            state = reduce_state_to_sliding_window(state,
                                                   distance_to_next_opponent,
                                                   min_sliding_window_size=self.min_sliding_window_size,
                                                   sliding_window_size_offset=self.sliding_window_size_offset)

        self.depth_first_iterative_deepening(state)
        return Action(self.move_to_make)


class LiveAgent(SlidingWindowVoronoiAgent):
    """
    Agent that plays the online game. The agent behind this is the SlidingWindowVoronoiAgent.
    """
    def __init__(self, model, pos, direction, speed=1, active=True, server_time_url="https://msoll.de/spe_ed_time"):
        super().__init__(model, pos, direction, speed, active)
        self.server_time_url = server_time_url

    def act(self, state):
        model = state_to_model(state)
        own_id = state["you"]
        _, _, is_endgame, min_player_ids = voronoi(model, own_id)
        if own_id in min_player_ids:
            min_player_ids.remove(own_id)
        if not is_endgame:
            pos = model.get_agent_by_id(own_id).pos
            opponent_pos = model.get_agent_by_id(min_player_ids[0]).pos
            distance_to_next_opponent = distance.euclidean(pos, opponent_pos)
            state = reduce_state_to_sliding_window(state,
                                                   distance_to_next_opponent,
                                                   min_sliding_window_size=self.min_sliding_window_size,
                                                   sliding_window_size_offset=self.sliding_window_size_offset)

        move = multiprocessing.Value('i', 4)
        reached_depth = multiprocessing.Value('i', 0)
        p = multiprocessing.Process(target=self.depth_first_iterative_deepening, name="DFID",
                                    args=(move, reached_depth, state))
        p.start()
        send_time = 1
        deadline = datetime.datetime.strptime(state["deadline"], "%Y-%m-%dT%H:%M:%SZ")
        response = requests.get(self.server_time_url)
        server_time = datetime.datetime.strptime(response.json()["time"], "%Y-%m-%dT%H:%M:%SZ")
        av_time = (deadline - server_time).total_seconds() - send_time
        p.join(av_time)

        # If thread is active
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()

        self.game_step += 1
        return Action(move.value)
