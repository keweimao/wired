# Install ipywidgets and IPython in they are not found
import ipywidgets as widgets
from IPython.display import display


# The game class to construct the tic tac toe game
class TicTacToe: 
    def __init__(self): 
        self.board = [ [' ' for col in range(3)] for row in range(3)]
        self.player = 'X'
        self.winner = ' '
        
        # User Interface element
        # Status Bar
        self.status = widgets.Label('Ready')
        # Cell Buttons
        self.buttons = [ [widgets.Button(description='') for button in range(3)] for row in range(3)]
        # register the make_move function to each button's onclick event
        for i in range(3): 
            for j in range(3): 
                self.buttons[i][j].on_click(self.make_move(i, j))
        self.button_list = [button for row in self.buttons for button in row]
        # Reset Button
        self.reset_button = widgets.Button(description='New Game', layout=widgets.Layout(width='450px'))
        self.reset_button.on_click(self.reset())
        # Output
        self.output = widgets.Output()
    
    # set a bot to play this game
    def set_bot(self, bot): 
        self.bot = bot
    
    # Start a NEW game
    def reset(self): 
        def on_reset_clicked(_): 
            self.start_over()
        return on_reset_clicked  
    
    # start a new game
    def start_over(self): 
        self.player = 'X'
        self.winner = ' '
        self.status.value = 'Ready'
        # clear the memory of 3x3 matrix
        for i in range(3): 
            for j in range(3): 
                self.board[i][j] = ' '
        # clear the buttons on the grid
        for button in self.button_list: 
            button.description = ' '
    
    # Put either "X" or "O" on the ith row and jth column
    def make_move(self, i, j): 
        def on_button_clicked(_): 
            # human move
            self.move(i,j)
            # bot move
            if self.winner==' ': 
                self.bot.move()
            
        return on_button_clicked
    
    # core function to make a move
    # for a human (click) or a bot
    def move(self, i, j):
        if self.winner==' ' and self.board[i][j] == ' ': 
            self.board[i][j] = self.player
            self.buttons[i][j].description = self.player

            # turn taking
            if self.player == 'X': 
                self.player = 'O'
            else: 
                self.player = 'X'
        self.status.value = 'In progress, ' + self.player + ' playing.'
        # check winner
        self.winner = self.check_win()
        if self.winner != ' ': 
            self.status.value = self.winner + ' won!'
    
    # Check if there is a winner
    # return the winner, 'X' or 'O'
    # OR, return ' ' if no winner
    def check_win(self): 
        # check diagnals
        if self.board[1][1]!=' ' and self.board[0][0]==self.board[1][1] and self.board[1][1]==self.board[2][2]:
            return self.board[0][0]
        if self.board[1][1]!=' ' and self.board[0][2]==self.board[1][1] and self.board[1][1]==self.board[2][0]:
            return self.board[1][1]
        
        # check rows
        for i in range(3): 
            if self.board[i][0]!=' ' and self.board[i][0]==self.board[i][1] and self.board[i][1]==self.board[i][2]:
                return self.board[i][0]
        
        # check columns
        for j in range(3): 
            if self.board[0][j]!=' ' and self.board[0][j]==self.board[1][j] and self.board[1][j]==self.board[2][j]:
                return self.board[0][j]
        
        # no winner found at this point
        return ' '
        
    
    def display(self): 
        self.grid = widgets.GridBox(self.button_list,layout=widgets.Layout( grid_template_columns="repeat(3, 150px)"))
        self.game_box = widgets.VBox([self.status, self.grid, self.reset_button])
        display(self.game_box, self.output)
        