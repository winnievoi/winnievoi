import sys
sys.path.append("../")
from DQN_model.RL_QG_agent import RL_QG_agent
board_size = 3
dqn = RL_QG_agent(board_size)