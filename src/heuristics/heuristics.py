from src.utils import Action, state_to_model, model_to_json, sync_voronoi, speed_one_voronoi
import numpy as np
from anytree import Node
import copy
import matplotlib.pyplot as plt


def multi_minimax(depth, game_state, super_pruning=False, use_voronoi=True):
    pruning_threshold = -1
    model = state_to_model(game_state)
    own_id = game_state["you"]
    start_node = Node(str(own_id))
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
        node = Node(str(min_player_ids), parent=start_node)
        min_move = float("inf")
        beta = float("inf")
        for opponent_id in min_player_ids:
            opponent = model.get_agent_by_id(opponent_id)

            min_move = min(min_move, minimax(max_player, opponent, depth-1, alpha, beta, False, model, node, use_voronoi))
            # for pre, fill, node in RenderTree(start_node):
            #     print("%s%s" % (pre, node.name))

            model = state_to_model(pre_state, trace_aware=True)
            max_player = model.get_agent_by_id(own_id)
            beta = min_move

            if alpha >= beta:
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


def minimax(max_player, min_player, depth, alpha, beta, is_max, model, last_parent_node, use_voronoi):
    if depth == 0 or not max_player.active or not model.running:
        return evaluate_position(model_to_json(model, trace_aware=True), max_player, min_player, depth, use_voronoi)
    if is_max:
        max_move = float("-inf")
        pre_state = model_to_json(model, trace_aware=True)
        for action in list(Action):
            update_game_state(model, max_player, action)
            node = Node(str(min_player.unique_id), parent=last_parent_node)

            max_move = max(max_move, minimax(max_player, min_player, depth-1, alpha, beta, False, model, node, use_voronoi))

            model = state_to_model(pre_state)
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
            node = Node(str(max_player.unique_id), parent=last_parent_node)

            min_move = min(min_move, minimax(max_player, min_player, depth-1, alpha, beta, True, model, node, use_voronoi))

            model = state_to_model(pre_state, trace_aware=True)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(beta, min_move)
            if alpha >= beta:
                break
        return min_move


def evaluate_position(state, max_player, min_player, depth, use_voronoi):
    if not use_voronoi:
        if max_player.active and not min_player.active:
            return float("inf")
        else:
            return -1 * depth
    else:
        model = state_to_model(state)
        if max_player.active and not min_player.active:
            return float("inf")
        elif not max_player.active:
            return -1000 * depth

        voronoi_cells, voronoi_counter = speed_one_voronoi(model)
        nb_cells = float(model.width * model.height)   # for normalization

        # voronoi region size comparison
        max_player_size = voronoi_counter[max_player.unique_id] if max_player.unique_id in voronoi_counter.keys() else 0
        min_player_size = voronoi_counter[min_player.unique_id] if min_player.unique_id in voronoi_counter.keys() else 0
        voronoi_region_points = (max_player_size - min_player_size) / nb_cells

        territory_bonus = 0
        for x in range(model.width):
            for y in range(model.height):
                if voronoi_cells[y, x, 0] == max_player.unique_id:
                    territory_bonus += add_territory_bonus(model, x, y)
        territory_bonus /= nb_cells

        return voronoi_region_points + territory_bonus


def add_territory_bonus(model, x, y):
    territory_bonus = 0.25
    bonus = 0
    # TODO: I guess you used try/except to avoid an IndexOutOfBoundsException, but I think that is incorrect.
    #       For example, non-surrounded cells at the right edge of the field get a bonus of 0 and non-surrounded cells
    #       at the top edge get a bonus of 0.6. However, all non-surrounded edge cells should get a bonus of 0.6, right?
    #       + x and y have to be swapped if used in model.cells (in this case only important if the field is
    #       non-quadratic)
    #       + I changed the territory_bonus to 0.25 so that all values are exactly in [0, 1]
    #       Checking for number under- or overflow should work fine.
    #       Suggestion:
    # directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # for d_x, d_y in directions:
    #     if 0 <= x + d_x < model.width and 0 <= y + d_y < model.height and model.cells[y + d_y, x + d_x] == 0:
    #         bonus += territory_bonus
    try:
        if model.cells[x+1, y] == 0:
            bonus += territory_bonus
        if model.cells[x, y+1] == 0:
            bonus += territory_bonus
        if model.cells[x-1, y] == 0:
            bonus += territory_bonus
        if model.cells[x, y-1] == 0:
            bonus += territory_bonus
    except:
        pass
    return bonus


def visualize_voronoi(cells):
    res = np.zeros((cells.shape[0], cells.shape[1]), dtype=np.int)
    for x in range(cells.shape[0]):
        for y in range(cells.shape[1]):
            res[x, y] = cells[x, y, 0]

    plt.imshow(res)
    plt.show()