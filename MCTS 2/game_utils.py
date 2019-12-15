import numpy as np

def get_valid_actions(game_state):
    '''
    return all possible action in current leaf state
    in:
    - leaf_state
    out:
    - set of possible actions ((row,col), action_idx)
    '''
    actions = []
    count = 0
    state_size = len(game_state)

    for i in range(state_size):
        for j in range(state_size):
            if game_state[i][j] == 0:
                actions.append([(i, j), count])
            count += 1
    return np.array(actions)
def is_terminal(game_state,win_mark):
    '''
    check terminal
    in: game state
    out: who wins? ('O', 'X', 'draw', None)
            (None = game not ended)
    '''
    def __who_wins(sums, win_mark):
        if np.any(sums == win_mark):
            return 1
        if np.any(sums == -win_mark):
            return 2
        return None

    def __is_terminal_in_condition(game_state, win_mark):
        # check row/col
        for axis in range(2):
            sums = np.sum(game_state, axis=axis)
            result = __who_wins(sums, win_mark)
            if result is not None:
                return result
        # check diagonal
        for order in [-1,1]:
            diags_sum = np.sum(np.diag(game_state[::order]))# order=1 means major diag while order=-1 means Deputy diag
            result = __who_wins(diags_sum, win_mark)
            if result is not None:
                return result
        return None

    n_rows = len(game_state)
    window_size = win_mark
    window_positions = range(n_rows - win_mark + 1)# default range(0,1)

    for row in window_positions:
        for col in window_positions:
            window = game_state[row:row+window_size, col:col+window_size]
            winner = __is_terminal_in_condition(window, win_mark)
            if winner is not None:
                return winner

    if not np.any(game_state == 0):
        '''
        no more action i can do
        '''
        return 3
    return None