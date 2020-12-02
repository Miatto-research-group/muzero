import numpy as np
from collections import deque
from .envs import Game
from scipy.spatial import distance

np.seed(1954)


class GateSynthesis(Game):
    num_players = 1
    num_actions = 7 #basic gates
    num_observations = 3 # number of states to pass to the representation network

    def __init__(self, target_unitary:np.array, max_rwd: int):
        super().__init__()
        #self.state = np.zeros((1, 3, 3), dtype=np.int) #???
        self.curr_space_state = 0 #???
        self.target_unitary = target_unitary
        self.max_rwd = max_rwd

    @property
    def turn(self): #no need for one player
        return 1 #always player 1's turn

    def step(self, gate: np.array):
        """
        Takes a step into the Hilbert space, applying a matrix
        """
        self.state = gate @ self.state ###on what?

    @property
    def reward(self) -> int:
        """
        The reward is as follows
        max_rwd upon reaching final target unitary
        +

        """
        # print("Reward ", int(self.end and (self.win_x or self.win_o)), flush=True)
        target_rwd = int(np.allclose(self.target_unitary, curr_space_state)) * self.max_rwd
        step_rwd =
        return target_rwd + step_rwd

    def distance(self) -> numeric:
        """
        Computes and returns the distance (as specified by the metric) between the
        current location of the agent and its target
        """
        new_shape = 1
        for s in len(self.curr_space_state.shape()) #for each dimension
            new_shape *= s
        reshaped_space_state = self.curr_space_state.reshape(new_shape)
        #fun part => https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        return np.abs(distance.directed_hausdorff(reshaped_space_state, self.target_unitary))

    ######################################"
    
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


    ########################################
    ########################################

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    S = np.array([[1, 0], [0, 1j]])
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    T = np.array([[1, 0], [0, np.exp((1j * np.pi) / 4)]])
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])




