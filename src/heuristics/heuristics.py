from src.utils import Action, state_to_model, model_to_json, sync_voronoi, speed_one_voronoi
import numpy as np
import copy
import matplotlib.pyplot as plt

end_game_depth = 5


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
        min_move = float("inf")
        beta = float("inf")
        for opponent_id in min_player_ids:
            opponent = model.get_agent_by_id(opponent_id)

            min_move = min(min_move, minimax(max_player, opponent, depth-1, max_depth, alpha, beta, False, model,
                                             use_voronoi, is_endgame))

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


def minimax(max_player, min_player, depth, max_depth, alpha, beta, is_max, model, use_voronoi, is_endgame):
    if depth == 0 or not max_player.active or not model.running:
        return evaluate_position(model_to_json(model, trace_aware=True), max_player, min_player, depth, max_depth,
                                 use_voronoi)
    if is_max or is_endgame:
        max_move = float("-inf")
        pre_state = model_to_json(model, trace_aware=True)
        for action in list(Action):
            update_game_state(model, max_player, action)

            max_move = max(max_move, minimax(max_player, min_player, depth-1, max_depth, alpha, beta, False, model,
                                             use_voronoi, is_endgame))

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

            min_move = min(min_move, minimax(max_player, min_player, depth-1, max_depth, alpha, beta, True, model,
                                             use_voronoi, is_endgame))

            model = state_to_model(pre_state, trace_aware=True)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(beta, min_move)
            if alpha >= beta:
                break
        return min_move


def evaluate_position(state, max_player, min_player, depth, max_depth, use_voronoi):
    if not use_voronoi:
        if max_player.active and not min_player.active:
            return float("inf")
        else:
            return -1 * depth
    else:
        # weights - all non-weighted evaluation values should be in [-1, 1]
        # using very large weight gaps is effectively like prioritization
        kill_weight = float("inf")
        death_weight = 1000
        voronoi_region_weight = 1
        territory_bonus_weight = 0.001  # only used to decide between positions with equal voronoi region evaluation

        model = state_to_model(state, trace_aware=True)
        if max_player.active and not min_player.active:
            # TODO: Maybe it is non-optimal to kill an opponent in a 1vx situation (its optimal in 1v1).
            #       It also makes a difference which opponent is killed if multiple opponents can be killed in a 1vx
            #       (e.g. kill the one with a larger voronoi region)
            return kill_weight
        elif not max_player.active:
            # TODO: Detect situations where the game is lost if we try to survive but we can force a kamikaze draw.
            #       At the moment the agent would always try to survive one more step and would only kamikaze if death
            #       is inevitable.
            death_eval = -depth
            if not min_player.active:
                # kamikaze is better than just dying without killing someone else
                death_eval += 0.1
            return death_eval / max_depth * death_weight

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

        return voronoi_region_points + territory_bonus


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