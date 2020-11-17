from src.utils import Action, state_to_model, model_to_json, Direction
import numpy as np
from anytree import Node
import copy


def multi_minimax(depth, game_state, super_pruning=True):
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

            min_move = min(min_move, minimax(max_player, opponent, depth-1, alpha, beta, False, model, node))
            # for pre, fill, node in RenderTree(start_node):
            #     print("%s%s" % (pre, node.name))

            model = state_to_model(pre_state, trace_aware=True)
            max_player = model.get_agent_by_id(own_id)
            if super_pruning and action == Action.CHANGE_NOTHING and min_move > pruning_threshold:
                return action
            beta = min_move

            if alpha >= beta:
                break
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


def minimax(max_player, min_player, depth, alpha, beta, is_max, model, last_parent_node):
    if depth == 0 or not max_player.active or not model.running:
        return evaluate_position(model, max_player, min_player, depth)
    if is_max:
        max_move = float("-inf")
        pre_state = model_to_json(model, trace_aware=True)
        for action in list(Action):
            update_game_state(model, max_player, action)
            node = Node(str(min_player.unique_id), parent=last_parent_node)

            max_move = max(max_move, minimax(max_player, min_player, depth-1, alpha, beta, False, model, node))

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

            min_move = min(min_move, minimax(max_player, min_player, depth-1, alpha, beta, True, model, node))

            model = state_to_model(pre_state, trace_aware=True)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(beta, min_move)
            if alpha >= beta:
                break
        return min_move


def evaluate_position(model, max_player, min_player, depth):
    if max_player.active and not min_player.active:
        return float("inf")
    else:
        return -1 * depth
