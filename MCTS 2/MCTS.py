from copy import deepcopy
import numpy as np
import time
from game_utils import *
class MCTS(object):
    def __init__(self,n_iterations=50, depth=15, exploration_constant=5.0, tree = None, win_mark=3, game_board=None, player=None,rl_model=None):
        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0
        self.rl_model = rl_model
        self.leaf_node_id = None

        n_rows = len(game_board)
        self.n_rows = n_rows
        self.win_mark = win_mark

        if tree == None:
            self.tree = self._set_tictactoe(game_board, player)
        else:
            self.tree = tree

    def _set_tictactoe(self, game_board, player):
        root_id = (0,)
        tree = {root_id: {'state': game_board,
                          'player': player,#current player
                          'child': [],#candidate child
                          'parent': None,
                          'n': 0,# num visited
                          'w': 0,# num wins
                          'q': None}}# q value
        return tree

    def selection(self):
        '''
        select leaf node which have maximum uct value
        in:
        - tree
        out:
        - leaf node id (node to expand)
        - depth (depth of node root=0)
        '''
        leaf_node_found = False
        leaf_node_id = (0,) # root node id

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['child'])
            

            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                maximum_uct_value = -200.0
                for i in range(n_child):
                    action = self.tree[node_id]['child'][i]

                    child_id = node_id + (action,)
                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']
                    total_n = self.total_n
                    

                    if n == 0:# have not visited
                        n = 1e-2
                        
                    # perform uct value
                    exploitation_value = w / n
                    exploration_value  = np.sqrt(np.log(total_n)/n)
                    uct_value = exploitation_value + self.exploration_constant * exploration_value

                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(leaf_node_id) # as node_id records selected action set
        
        return leaf_node_id, depth

    def expansion(self, leaf_node_id):
        '''
        create all possible outcomes from leaf node
        in: tree, leaf_node
        out: expanded tree (self.tree),
             randomly selected child node id (child_node_id)
        '''
        leaf_state = self.tree[leaf_node_id]['state']
        winner = self._is_terminal(leaf_state)
        possible_actions = self._get_valid_actions(leaf_state)
        child_node_id = leaf_node_id # default value
        
        if winner is None:
            
            


            '''
            set child_id for each child 
            '''
            childs = []
            for action_set in possible_actions:
                action, action_idx = action_set
                state = deepcopy(self.tree[leaf_node_id]['state'])# when simulation end retreat to current node
                current_player = self.tree[leaf_node_id]['player']

                if current_player == 'O':
                    next_turn = 'X'
                    state[action] = 1
                else:
                    next_turn = 'O'
                    state[action] = -1

                child_id = leaf_node_id + (action_idx, )
                childs.append(child_id)
                # save the child state
                self.tree[child_id] = {'state': state,
                                    'player': next_turn,
                                    'child': [],
                                    'parent': leaf_node_id,
                                    'n': 0, 'w': 0, 'q':0}
                self.tree[leaf_node_id]['child'].append(action_idx)# use action_idx represent child_id
            if self.rl_model:
                '''
                rl give a prior action
                '''
                possible_actions = np.array(possible_actions)[:,1].tolist()
                
                prioritize_action = self.rl_model.place(leaf_state,possible_actions)
                
                child_node_id = leaf_node_id + (prioritize_action, )
            else:
                rand_idx = np.random.randint(low=0, high=len(childs), size=1)
                child_node_id = childs[rand_idx[0]]
        return child_node_id

    def _is_terminal(self,leaf_state):
        return is_terminal(leaf_state,self.win_mark)

    
    def _get_valid_actions(self,leaf_state):
        return get_valid_actions(leaf_state)
    def simulation(self, child_node_id):
        '''
        simulate game from child node's state until it reaches the resulting state of the game.
        in:
        - child node id (randomly selected child node id from `expansion`)
        out:
        - winner ('O', 'X', 'draw')=>(1,2,3)
        '''
        self.total_n += 1
        state = deepcopy(self.tree[child_node_id]['state'])# save current state
        previous_player = deepcopy(self.tree[child_node_id]['player'])
        anybody_win = False

        while not anybody_win:
            winner = self._is_terminal(state)
            if winner is not None:
                
                anybody_win = True
            else:
                possible_actions = self._get_valid_actions(state)
                # randomly choose action for simulation (= random rollout policy)
                rand_idx = np.random.randint(low=0, high=len(possible_actions), size=1)[0]
                action, _ = possible_actions[rand_idx]

                if previous_player == 'O':
                    
                    state[action] = -1
                    previous_player = 'X'
                else:
                    
                    state[action] = 1
                    previous_player = 'O'

        return winner

    def backprop(self, child_node_id, winner):
        # player = deepcopy(self.tree[(0,)]['player'])

        if winner == 3:
            reward = 0
        elif winner == 1:
            reward = 1
        else:
            reward = -1

#         finish_backprob = False
        node_id = child_node_id
        while node_id:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']

            node_id = self.tree[node_id]['parent']

    def solve(self):
        for i in range(self.n_iterations):# for simulation
            leaf_node_id, depth_searched = self.selection()
            child_node_id = self.expansion(leaf_node_id)# choose a chile node randomly
            
            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)
            if depth_searched > self.depth:
                break

        # SELECT BEST ACTION
        current_state_node_id = (0,)
        action_candidates = self.tree[current_state_node_id]['child']
        # qs = [self.tree[(0,)+(a,)]['q'] for a in action_candidates]
        best_q = -100
        for a in action_candidates:
            q = self.tree[(0,a)]['q']
            if q > best_q:
                best_q = q
                best_action = a

        

        return best_action, best_q, depth_searched+1