import numpy as np
import matplotlib.pyplot as plt
from Game import Game
from MCTS import MCTS
from DQN_model.RL_QG_agent import RL_QG_agent
import tensorflow as tf
def evaluate_mcts_without_rl(max_game_iter,board_size):
    env = Game(board_size,is_random_policy=True)

    game_iter = 0
    agent_win = 0
    human_win = 0
    draw = 0

    node_checked = 0
    while game_iter<max_game_iter:
        action = -1
        if env.turn == 0:
            mcts = MCTS(n_iterations=50, depth=10, exploration_constant=10, 
            game_board=env.game_state,win_mark=env.board_size, player='O')
            action, best_q, depth = mcts.solve()
            node_checked+=depth

        _,win_flag = env.step(action)
        
        if win_flag:
            if win_flag==1:
                print('agent win')
                agent_win+=1
            elif win_flag==2:
                human_win+=1
                print('human win')
            else:
                draw+=1
                print('draw')
            env.reset()
            game_iter+=1
    win_rate = agent_win/(max_game_iter)
    avg_node_checked = node_checked/(max_game_iter)
    print('win_rate with boardsize '+str(board_size)+': ',win_rate)
    print('average_nodes checked :',int(avg_node_checked))
    return win_rate,avg_node_checked


def evaluate_mcts_with_rl(max_game_iter,board_size):

    env = Game(board_size,is_random_policy=True)

    game_iter = 0
    agent_win = 0
    human_win = 0
    draw = 0

    rl_model = RL_QG_agent(board_size)
    node_checked = 0
    while game_iter<max_game_iter:
        
        action = -1
        if env.turn == 0:
            mcts = MCTS(n_iterations=50, depth=10, exploration_constant=10, 
            game_board=env.game_state,win_mark=env.board_size, player='O',rl_model=rl_model)
            action, best_q, depth = mcts.solve()
            node_checked+=depth

        _,win_flag = env.step(action)
        
        if win_flag:
            if win_flag==1:
                print('agent win')
                agent_win+=1
            elif win_flag==2:
                human_win+=1
                print('human win')
            else:
                draw+=1
                print('draw')
            env.reset()
            game_iter+=1
    win_rate = agent_win/(max_game_iter)
    avg_node_checked = node_checked/(max_game_iter)
    print('win_rate with boardsize '+str(board_size)+': ',win_rate)
    print('average_nodes checked :',int(avg_node_checked))
    return win_rate,avg_node_checked


if __name__ == '__main__':
    win_rate_set = []
    node_checked_set = []
    win_rate_set_rl = []
    node_checked_set_rl = []
    max_game_iter = 100
    experiments_board = [3,4,5,6,7]
    for board_size in experiments_board:
        tf.reset_default_graph()
        win_rate,avg_node_checked = evaluate_mcts_without_rl(max_game_iter,board_size)
        win_rate_set.append(win_rate)
        node_checked_set.append(avg_node_checked)

    for board_size in experiments_board:
        tf.reset_default_graph()
        win_rate,avg_node_checked = evaluate_mcts_with_rl(max_game_iter,board_size)
        win_rate_set_rl.append(win_rate)
        node_checked_set_rl.append(avg_node_checked)
    width = 0.3
    x_ = [x+width for x  in experiments_board]

    plt.bar(x=experiments_board,height=win_rate_set,width=width,color='r',label = 'with_out_rl_winrate')
    plt.bar(x=x_,height=win_rate_set_rl,width=width,color='g',label = 'with_rl')
    plt.xlabel('board_size')
    plt.ylabel('win_rate')
    plt.legend()
    plt.savefig('win_rate.png')
    plt.show()

    # x_2 = [i+width for i in x_1]
    plt.bar(x=experiments_board,height=node_checked_set,width=width,color='r',label = 'with_out_rl_efficiency')
    plt.bar(x=x_,height=node_checked_set_rl,width=width,color='g',label = 'with_rl')
    plt.xlabel('board_size')
    plt.ylabel('nodes_checked')
    plt.legend()
    plt.savefig('node_checked.png')
    plt.show()

