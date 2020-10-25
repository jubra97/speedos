from src.utils import Action


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


def update_game_state(game_state, child):
    ...


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


def evaluate_position(model, agent):
    return 1 if agent.active else -1


