from src.utils import Action, state_to_model, model_to_json, sync_voronoi, speed_one_voronoi, hash_state
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

max_cache_depth = 4


def multi_minimax_depth_first_iterative_deepening(shared_move_var, shared_dep, game_state, super_pruning=False, use_voronoi=True):
    depth = 1
    globals()["cache"] = dict() #defaultdict(int)

    while True:  # canceled from outside
        move_to_make = multi_minimax(depth, game_state, super_pruning=super_pruning, use_voronoi=use_voronoi)
        shared_move_var.value = move_to_make.value
        shared_dep.value = depth
        depth += 1


def multi_minimax(depth, game_state, super_pruning=False, use_voronoi=True):
    max_depth = depth
    pruning_threshold = -1
    model = state_to_model(game_state)
    own_id = game_state["you"]
    _, _, is_endgame = speed_one_voronoi(model, own_id)
    max_player = model.get_agent_by_id(own_id)
    min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
    min_player_ids.remove(own_id)

    move_to_make = Action.CHANGE_NOTHING
    max_move = float("-inf")
    alpha = float("-inf")
    copy_actions = copy.deepcopy(list(Action))
    for action in reversed(copy_actions):
        pre_state = model_to_json(model, trace_aware=True)
        update_game_state(model, max_player, action)
        tree_path = str(action.value)
        min_move = float("inf")
        beta = float("inf")
        for opponent_id in min_player_ids:
            opponent = model.get_agent_by_id(opponent_id)

            min_move = min(min_move, minimax(max_player, opponent, depth-1, max_depth, alpha, beta, False, model,
                                             use_voronoi, is_endgame, tree_path=tree_path))

            model = state_to_model(pre_state, trace_aware=True)
            max_player = model.get_agent_by_id(own_id)
            beta = min_move

            if alpha >= beta:
                break
            if is_endgame:
                break

        if super_pruning and action == Action.CHANGE_NOTHING and min_move > pruning_threshold:
            return action

        if min_move >= max_move:
            if min_move == max_move:
                move_to_make = np.random.choice([move_to_make, action])
            else:
                move_to_make = action
            max_move = min_move
            alpha = max_move
    return move_to_make


def update_game_state(model, agent, action):
    agent.action = action
    model.step_specific_agent(agent)


def minimax(max_player, min_player, depth, max_depth, alpha, beta, is_max, model, use_voronoi, is_endgame, tree_path=None):
    if depth == 0 or not max_player.active or not model.running:
        return evaluate_position(model, max_player, min_player, depth, max_depth,
                                 use_voronoi, tree_path=tree_path)
    if is_max or is_endgame:
        max_move = float("-inf")
        pre_state = model_to_json(model, trace_aware=True)
        for action in list(Action):
            update_game_state(model, max_player, action)
            tree_path += str(action.value)

            max_move = max(max_move, minimax(max_player, min_player, depth-1, max_depth, alpha, beta, False, model,
                                             use_voronoi, is_endgame, tree_path=tree_path))

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
        for action in list(Action):
            update_game_state(model, min_player, action)
            tree_path += str(action.value)

            min_move = min(min_move, minimax(max_player, min_player, depth-1, max_depth, alpha, beta, True, model,
                                             use_voronoi, is_endgame, tree_path=tree_path))

            model = state_to_model(pre_state, trace_aware=True)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(beta, min_move)
            if alpha >= beta:
                break
        return min_move


def evaluate_position(model, max_player, min_player, depth, max_depth, use_voronoi, tree_path=None, caching_enabled=False):
    # use cached value
    #print("evaluate " + tree_path)
    if caching_enabled and globals()["cache"] is not None:
        cache_key = tree_path #hash_state(state)
        if cache_key in globals()["cache"]:
            print("found cached " + tree_path)
            return globals()["cache"][cache_key]

    if not use_voronoi:
        if max_player.active and not min_player.active:
            return float("inf")
        else:
            # subtract a high number if agent died (otherwise dying in the last step is as good as survival)
            if not max_player.active:
                death_penalty = 1000
            else:
                death_penalty = 0
            return -1 * depth - death_penalty
    else:
        # weights - all non-weighted evaluation values should be in [-1, 1]
        # using very large weight gaps is effectively like prioritization
        kill_weight = float("inf")
        death_weight = 1000
        voronoi_region_weight = 1
        territory_bonus_weight = 0.001  # only used to decide between positions with equal voronoi region evaluation

        # TODO: Bonus points for adjacent battlezones

        if max_player.active and not min_player.active:
            utility = kill_weight
            if caching_enabled and depth < max_cache_depth and globals()["cache"] is not None:
                globals()["cache"][cache_key] = utility  # cache result
            return utility
        elif not max_player.active:
            # TODO: Detect situations where the game is lost if we try to survive but we can force a kamikaze draw.
            #       At the moment the agent would always try to survive one more step and would only kamikaze if death
            #       is inevitable.
            death_eval = -depth - 1     # -1 to keep negative when depth is 0
            if not min_player.active:
                # kamikaze is better than just dying without killing someone else
                death_eval += 0.1
            utility = death_eval / max_depth * death_weight
            if caching_enabled and depth < max_cache_depth and globals()["cache"] is not None:
                globals()["cache"][cache_key] = utility  # cache result
            return utility

        voronoi_cells, voronoi_counter, is_endgame = speed_one_voronoi(model, max_player.unique_id)
        nb_cells = float(model.width * model.height)   # for normalization

        # voronoi region size comparison
        max_player_size = voronoi_counter[max_player.unique_id] if max_player.unique_id in voronoi_counter.keys() else 0
        min_player_size = voronoi_counter[min_player.unique_id] if min_player.unique_id in voronoi_counter.keys() else 0
        voronoi_region_points = (max_player_size - min_player_size) / nb_cells * voronoi_region_weight

        territory_bonus = 0
        for x in range(model.width):
            for y in range(model.height):
                if voronoi_cells[y, x, 0] == max_player.unique_id:
                    territory_bonus += add_territory_bonus(model, x, y)
        territory_bonus *= territory_bonus_weight / nb_cells

        utility = voronoi_region_points + territory_bonus
        if caching_enabled and depth < max_cache_depth and globals()["cache"] is not None:
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


def visualize_voronoi(cells):
    res = np.zeros((cells.shape[0], cells.shape[1]), dtype=np.int)
    for x in range(cells.shape[0]):
        for y in range(cells.shape[1]):
            res[x, y] = cells[x, y, 0]

    plt.imshow(res)
    plt.show()
