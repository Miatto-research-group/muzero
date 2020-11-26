import numpy as np
from collections import deque
from .envs import Game

class TicTacToe(Game):
    num_players = 2
    num_actions = 9
    num_observations = 3 # number of states to pass to the representation network

    def __init__(self):
        super().__init__()
        self.state = np.zeros((2, 3, 3), dtype=np.int)

    @property
    def turn(self): #no need for one player
        return np.sum(self.state) % 2
    
    def play(self, action) -> int: #drastic change
        if type(action) is int:
            action = self.action(action)
        if not self.valid_action(action):
            self.show
            raise ValueError(f"invalid action \n{action}")
        self.state[self.turn] += action
        #print("Play ", self.reward, "\n",  self.state[self.turn], flush=True)
        return self.reward
    
    def action(self, action_index: int) -> np.array: #drastic change
        z = np.zeros(9, dtype=np.int)
        z[action_index] = 1
        return z.reshape((3, 3))

    @property
    def end(self):
        return self.win_x or self.win_o or self.draw

    @property
    def reward(self) -> int:
        #print("Reward ", int(self.end and (self.win_x or self.win_o)), flush=True)
        return int(self.end and (self.win_x or self.win_o))

    @property
    def mask(self):
        return 1 - (self.state[0] + self.state[1]).reshape(-1)

    def valid_action(self, action):
        return action.reshape(-1)@self.mask > 0
    
    @staticmethod
    def check_win(board):
        vertical = np.isclose(np.sum(board, axis=0), 3).any()
        horizontal = np.isclose(np.sum(board, axis=1), 3).any()
        diagonals = np.isclose([np.trace(board), np.trace(np.rot90(board))], 3).any()
        #print("Check Win: ", vertical, horizontal, diagonals, flush=True)
        return vertical or horizontal or diagonals

    @property
    def win_x(self):
        return self.check_win(self.state[0])
        
    @property
    def win_o(self):
        return self.check_win(self.state[1])
    
    @property
    def draw(self):
        return np.allclose(self.state[0] + self.state[1], 1)

    @property
    def show(self):
        squares = [" ", " ", " ", " ", " ", " ", " ", " ", " "]
        for k in range(9):
            if self.state[0].reshape(-1)[k] == 1: squares[k] = "x"
            if self.state[1].reshape(-1)[k] == 1: squares[k] = "o"
        print(" ___\n|"+"".join(squares[:3]) + "|\n|" + "".join(squares[3:6]) + "|\n|" + "".join(squares[6:])+"|\n ---", flush=True)
