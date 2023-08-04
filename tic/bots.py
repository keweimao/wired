import random
import numpy as np

class RandomBot:
    def __init__(self, game, player): 
        if player not in ['X', 'O']:
            raise ValueError("Player must be either X or O!")
        self.game = game
        self.player = player
    
    # a move function to pick a random cell
    def move(self): 
        avail_cells = [ (i,j) for i in range(3) for j in range(3) if self.game.board[i][j]==' ' ]
        cell = random.choice(avail_cells)
        self.game.move(cell[0], cell[1])

# a bot capable of reinforcement learning (Q-Learning)
# for playing tic-tac-toe
class RLBot: 
    def __init__(self, game, player, learning_rate=0.5, discount_factor=0.9, exploration=0.1): 
        self.game = game
        self.player = player
        self.q_table = {}
        self.set_params(learning_rate, discount_factor, exploration)
    
    def set_params(self, learning_rate=0.5, discount_factor=0.9, exploration=0.1): 
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration
    
    def change_game(self, game): 
        self.game = game
    
    # get the current state of the board
    # which is a 2-dimensional np.array of 'X', 'O', and ' ' values
    def get_state(self, board): 
        return str(board.reshape(-1))
    
    # get q values for a given state
    def get_q_values(self, state): 
        if state not in self.q_table: 
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]
    
    # get available actions (empty cells)
    def get_avail_actions(self, board): 
        actions_2d = list(zip(*np.where(board==' ')))
        actions_1d = [ np.ravel_multi_index(act, (3,3)) for act in actions_2d ]
        return actions_1d
    
    # check if the board is full (a draw)
    def board_is_full(self, board): 
        return len(self.get_avail_actions(board))==0
    
    # pick an action (move)
    # given a state and avail actions
    def select_action(self, state, avail_actions): 
        # random move
        if np.random.random() < self.exploration_rate: 
            action = np.random.choice(avail_actions)
        # best move based on past learning (best q value)
        else: 
            q_values = self.get_q_values(state)
            avail_q_values = q_values[avail_actions]
            best_action_index = np.argmax(avail_q_values)
            action = avail_actions[best_action_index]
        return action
    
    # update q values for old_state => action => new_state
    # with reward (positive reward or negative penalty)
    def update_q_values(self, old_state, action, reward, new_state): 
        old_q_values = self.get_q_values(old_state)
        new_q_values = self.get_q_values(new_state)
        old_q_values[action] = old_q_values[action] + self.learning_rate * (reward + self.discount_factor * np.max(new_q_values)  - old_q_values[action])
        
    # make a move
    def move(self): 
        board = np.array(self.game.board)
        state = self.get_state(board)
        avail_actions = self.get_avail_actions(board)
        action_1d = self.select_action(state, avail_actions)
        # action_2d will have (row, col) indices
        action_2d = np.unravel_index(action_1d, (3,3))
        i = action_2d[0]
        j = action_2d[1]
        self.game.move(i, j)
        new_board = np.array(self.game.board)
        new_state = self.get_state(new_board)
        # return state (old_state), action, new_state
        return state, action_1d, new_state
    