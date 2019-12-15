import sys
sys.path.append('../')
from Game import Game
from RL_QG_agent import RL_QG_agent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
for board_size in [3,4,5,6,7]:

    
    tf.reset_default_graph()

    env = Game(board_size,True)

    game_iter = 50
    

    dqn = RL_QG_agent(env.board_size)
    field = board_size**2
    ob = env.reset()
    done = False

    agent_win = 0
    human_win = 0
    draw = 0
    
    losses=[]
    
    # avg_rewards = []
    reward = 0
    learning_step = 0
    while game_iter>0 or learning_step < 100:

        
        sum_agent = 0
        # rewards = []
        
            
        done = False
        env.reset()
        while not done:
            action = dqn.place(ob,env.valid_position())
            ob_,flag = env.step(action)
            sum_agent+=1
            reward = 0.05*(sum_agent/field)
            if flag:
                done = True
                if flag==1:
                    agent_win+=1
                    print('agent win')
                    reward = 1
                else:
                    draw+=1
                    reward = 0.5
                    print('draw')

            dqn.step_counter += 1

            dqn.store_transition(ob,action, reward, ob_, done)

            if(dqn.step_counter>128):
                learning_step+=1
                losses.append(dqn.learn())

    
            if not done:

                action = -1
                ob_,flag = env.step(action)

                if flag:
                    done = True
                    if flag==2:
                        print('human win')
                        reward = -1
                        human_win+=1
                    else:
                        reward = -0.5
                        draw+=1
                        print('draw')
                    dqn.store_transition(ob,action, reward, ob_, done)
            # rewards.append(reward)
#         /print(rewards)
                
       
        # avg_rewards.append(np.array(rewards).mean())

        game_iter-=1
    
    # plt.plot(list(range(len(avg_rewards))),avg_rewards)
    # plt.savefig(str(board_size)+'_board_size_reward_cruve.png')
    # plt.xlabel('game_iter')
    # plt.ylabel('mean_reward')
    # plt.show()
    plt.plot(list(range(len(losses))),losses)
    plt.xlabel('learning_step')
    plt.ylabel('loss')
    plt.savefig(str(board_size)+'learning_loss_cruve.png')
    
    plt.show()