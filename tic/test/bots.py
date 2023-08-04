import random
import numpy as np

# random bot
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
        
# Reinforcement Learning bot
class RLBot:
    def __init__(self, game, player, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.1):
        self.game = game
        self.player = player
        self.q_table = {}
        self.set_params(learning_rate, discount_factor, exploration_rate)
    
    # change game
    def change_game(self, game): 
        self.game = game
    
    # parameters affecting the bot's learning behavior
    def set_params(self, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.1): 
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
    # board is a 2-dimensional np.array of 'X', 'O', or ' ' values
    def get_state(self, board):
        # board = np.array(self.game.board)
        return str(board.reshape(-1))  # flatten to 1-dimensional
        
    # get current q values of a given `state`
    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]
    
    # get available actions
    def get_avail_actions(self, board): 
        actions_2d = list(zip(*np.where(board == ' ')))
        actions_1d = [np.ravel_multi_index(act, (3, 3)) for act in actions_2d]
        return actions_1d
    
    # a potential draw when board is full
    def board_is_full(self, board): 
        return len(self.get_avail_actions(board))==0
    
    # select an action given a `state` and `available actions`
    def select_action(self, state, available_actions):
        if np.random.random() < self.exploration_rate:
            action = np.random.choice(available_actions)
        else: 
            q_values = self.get_q_values(state)
            # Get the q_values of the available actions
            available_q_values = q_values[available_actions]
            # select the action with the highest available q value
            best_action_index = np.argmax(available_q_values)
            # return to available_actions to find the original index for the best value
            action = available_actions[best_action_index]
        return action
    
    # update q values for the (old) state and action
    # based on 1) immediate reward, and 2) existing reward in the new state (that the action leads to)
    def update_q_values(self, old_state, action, reward, new_state):
        old_q_values = self.get_q_values(old_state)
        new_q_values = self.get_q_values(new_state)
        old_q_values[action] = old_q_values[action] + self.learning_rate * (reward + self.discount_factor * np.max(new_q_values) - old_q_values[action])
        
    # make a move, an exploratory (random) or learned one. 
    # return [old_state, action, new_state]
    def move(self):
        board = np.array(self.game.board)
        state = self.get_state(board)
        actions_1d = self.get_avail_actions(board)
        action_1d = self.select_action(state, actions_1d)
        action_2d = np.unravel_index(action_1d, (3,3))
        # print(action_1d, action_2d)
        self.game.move(action_2d[0], action_2d[1])
        new_board = np.array(self.game.board)
        new_state = self.get_state(new_board)
        return state, action_1d, new_state