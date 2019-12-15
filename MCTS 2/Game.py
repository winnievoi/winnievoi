import numpy as np
import copy
from RL_QG_agent import RL_QG_agent
from game_utils import get_valid_actions,is_terminal
from MCTS import MCTS
import time

class Game(object):
    def __init__(self,board_size,is_random_policy):
        self.board_size = board_size
        
        self.win_num = self.board_size
        self.is_random_policy = is_random_policy
        self.to_draw = {1:'O',-1:'X',0:' '}
        self.reset()
        
    def reset(self,):
        # random choose who play first(my be hard)
        self.game_state = np.zeros((self.board_size,self.board_size))
        self.win_flag = 0 # 1->agent,2->human,3->draw,0->playing
#         self.board_action = np.zeros(board_size**2)
        self.place_num=0
        self.turn = 0
            
        return self.game_state
            
    def valid_position(self):
        
        return get_valid_actions(self.game_state)[:,1].tolist()
    def step(self,action):
        if action==-1:#human play
            valid_pos = self.valid_position()
            if self.is_random_policy:
                action = np.random.choice(valid_pos,size=1)[0]
            else:
                while action not in valid_pos:
                   action = int(input('select a position to place your piece'+str(valid_pos)))
             
        row = action//self.board_size
        col = action%self.board_size
        self.game_state[row,col] =1 if self.turn==0 else -1
        
        self.turn = abs(self.turn-1)
        self.place_num+=1
        
        self.win_flag = self.check_win()
        
        if not self.is_random_policy:self.render()
        return self.game_state,self.win_flag
        
    def check_win(self):
        return is_terminal(self.game_state,self.win_num)
    
    

    

    
    def render(self):
        for i in range(self.board_size):
            for j in range(self.board_size-1):
                print(self.to_draw[self.game_state[i,j]],end=' |')
            print(self.to_draw[self.game_state[i,-1]])
            if i< self.board_size -1:print('---'*(self.board_size))