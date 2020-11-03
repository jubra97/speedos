from src.utils import Action, state_to_model, model_to_json
import numpy as np


def multi_minimax(depth, game_state):
    model = state_to_model(game_state)
    own_id = game_state["you"]
    max_player = model.get_agent_by_id(own_id)
    min_player_ids = list(map(lambda a: a.unique_id, model.active_speed_agents))
    min_player_ids.remove(own_id)

    move_to_make = Action.CHANGE_NOTHING
    max_move = float("-inf")
    alpha = float("-inf")
    for action in list(Action):
        pre_state = model_to_json(model)
        update_game_state(model, max_player, action)
        min_move = float("inf")
        beta = float("inf")
        for opponent_id in min_player_ids:
            opponent = model.get_agent_by_id(opponent_id)

            min_move = min(min_move, minimax(max_player, opponent, depth-1, alpha, beta, False, model))

            model = state_to_model(pre_state)
            max_player = model.get_agent_by_id(own_id)

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


def minimax(max_player, min_player, depth, alpha, beta, is_max, model):
    # TODO: Check if there is no legal (non-loosing) move instead of checking whether the game is already lost/over
    if depth == 0 or not max_player.active or not model.running:
        return evaluate_position(model, max_player, depth)
    if is_max:
        max_move = float("-inf")
        pre_state = model_to_json(model)
        for action in list(Action):
            update_game_state(model, max_player, action)

            max_move = max(max_move, minimax(max_player, min_player, depth-1, alpha, beta, False, model))

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
        pre_state = model_to_json(model)
        for action in list(Action):
            update_game_state(model, min_player, action)

            min_move = min(min_move, minimax(max_player, min_player, depth-1, alpha, beta, True, model))

            model = state_to_model(pre_state)
            own_id = max_player.unique_id
            max_player = model.get_agent_by_id(own_id)
            min_player_id = min_player.unique_id
            min_player = model.get_agent_by_id(min_player_id)

            beta = min(alpha, min_move)
            if alpha >= beta:
                break
        return min_move


def evaluate_position(model, agent, depth):
    return 1 if agent.active else -1 * depth


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



