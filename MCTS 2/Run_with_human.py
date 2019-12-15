from Game import Game
from MCTS import MCTS
def main(board_size):
    env = Game(board_size,is_random_policy=False)
    done = False
    while not done:
        
        mcts = MCTS(n_iterations=50, depth=10, exploration_constant=10, 
        game_board=env.game_state,win_mark=env.board_size, player='O')
        action, best_q, depth = mcts.solve()
        _,win_flag = env.step(action)

        if win_flag:
            done = True
            if win_flag==1:
                print('agent win')
                
            elif win_flag==2:
                
                print('human win')
            else:
                
                print('draw')
        if not done:
            #human turn

            _,win_flag = env.step(-1)
            if win_flag:
                done = True
                if win_flag==1:
                    print('agent win')
                    
                elif win_flag==2:
                    
                    print('human win')
                else:
                    
                    print('draw')

if __name__ == "__main__":
    main(3)

