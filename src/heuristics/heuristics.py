from src.utils import Action, state_to_model, model_to_json
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


def multi_minimax(depth, game_state):
    threshold_value = -depth+depth/2
    best_value = 0
    model = state_to_model(game_state)
    own_id = game_state["you"]
    start_node = Node(str(own_id))
    max_player = model.get_agent_by_id(own_id)
    min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
    min_player_ids.remove(own_id)

    move_to_make = Action.CHANGE_NOTHING
    max_move = float("-inf")
    alpha = float("-inf")
    for action in reversed(list(Action)):
        pre_state = model_to_json(model)
        update_game_state(model, max_player, action)
        node = Node(str(min_player_ids), parent=start_node)
        min_move = float("inf")
        beta = float("inf")
        for opponent_id in min_player_ids:
            opponent = model.get_agent_by_id(opponent_id)

            min_move = min(min_move, minimax(max_player, opponent, depth-1, alpha, beta, False, model, node))
            # for pre, fill, node in RenderTree(start_node):
            #     print("%s%s" % (pre, node.name))

            model = state_to_model(pre_state)
            # BUG: min_move is almost every time inf for change nothing: WHY?
            max_player = model.get_agent_by_id(own_id)
            if action == Action.CHANGE_NOTHING and min_move > -1:
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
        pre_state = model_to_json(model)
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
            #if alpha >= beta:
            #    break
        return max_move
    else:
        min_move = float("inf")
        pre_state = model_to_json(model)
        for action in list(Action):
            update_game_state(model, min_player, action)
            node = Node(str(max_player.unique_id), parent=last_parent_node)

            min_move = min(min_move, minimax(max_player, min_player, depth-1, alpha, beta, True, model, node))

            model = state_to_model(pre_state)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(alpha, min_move)
            #if alpha >= beta:
            #    break
        return min_move


def check_legal_move_left(agent_id, state):
    for action in list(Action):
        model = state_to_model(state)
        agent = model.get_agent_by_id(agent_id)
        update_game_state(model, agent, action)
        if agent.active:
            return True
    return False


def evaluate_position(model, max_player, min_player, depth):
    """if check_legal_move_left(max_player.unique_id, model_to_json(model)) and not \
            check_legal_move_left(min_player.unique_id, model_to_json(model)):"""
    if max_player.active and not min_player.active:
        return float("inf")
    else:
        return -1 * depth



"""all_values = 0
counter = 0
distance = 5
agent_pos = max_player.pos
x_start = agent_pos[0] - distance
y_start = agent_pos[1] - distance
x_end = agent_pos[0] + distance
y_end = agent_pos[1] + distance
for x in range(x_start, x_end):
    for y in range(y_start, y_end):
        try:
            if model.cells[x][y] == 0:
                pass
            else:
                all_values += 1
        except IndexError:
            all_values += 1
        counter += 1
density = all_values / counter"""

"""
def minimax(max_player, min_player, depth, alpha, beta, is_max, game_state):
    # if depth == 0: OR NO MORE VALID MOVES
    # return heuristic value
    if is_max:
        max_move = float("-inf")
        for child in max_player:
            update_game_state(game_state, child)
            max_move = max(max_move, minimax(child, min_player, depth-1, alpha, beta, False, game_state))
            alpha = max(alpha, max_move)
            if alpha >= beta:
                break
        return max_move
    else:
        min_move = float("inf")
        for child in min_player:
            update_game_state(game_state, child)
            min_move = min(min_move, minimax(max_player, child, depth-1, alpha, beta, True, game_state))
            beta = min(alpha, min_move)
            if alpha >= beta:
                break
        return min_move
"""

"""
def multi_minimax(max_player, min_players, depth, game_state):
    move_to_make = 0
    max_move = float("-inf")
    alpha = float("-inf")
    for child in max_player:
        update_game_state(game_state, child)
        min_move = float("-inf")
        beta = float("inf")
        for opponent in min_players:
            min_move = min(min_move, minimax(child, opponent, depth-1, alpha, beta, False, game_state))
            beta = min_move
            if alpha >= beta:
                break
        if min_move >= max_move:
            move_to_make = child
            max_move = min_move
            alpha = max_move
    return move_to_make
"""

