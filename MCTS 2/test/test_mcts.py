import sys
sys.path.append("../")
from MCTS import MCTS
import numpy as np
game_state = np.zeros((3,3))
game_state[0,0] =1
mcts = MCTS(n_iterations=255, depth=10, exploration_constant=10, 
            game_board=game_state,win_mark=3, player='O')

print(mcts._is_terminal(game_state))
# leaf_node_id,depth = mcts.selection()
# print(leaf_node_id,depth)
# child_node_id = mcts.expansion(leaf_node_id)
# print(child_node_id)
# winner = mcts.simulation(child_node_id)
# print(winner)
# mcts.backprop(child_node_id,winner)


