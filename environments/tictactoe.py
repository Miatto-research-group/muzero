import numpy as np
from collections import deque

class TicTacToe:
    num_observations = 3
    MDP = False

    def __init__(self):
        self.board = np.zeros((2, 3, 3), dtype=np.int)
        self.observations = deque(list(np.zeros((self.num_observations, 2, 3, 3), dtype=np.int)), maxlen=self.num_observations)
        self.end = False
        self.actions = []
        # self.total = 0
        self.record_observation()

    def from_observations(self, observations):
        self.observations = observations
        self.board = observations[-1].copy()
        self.end = self.win_o or self.win_x or self.draw
        # self.total = np.sum(self.board)

    @property
    def turn(self):
        return np.sum(self.board) % 2

    def record_observation(self):
        self.observations.append(self.board.copy())

    def play(self, action) -> int:
        if type(action) is int:
            action = self.action(action)
        if not self.valid_action(action):
            raise ValueError(f"invalid action {action} with board {self.board} and mask {self.mask}")
        self.board[self.turn] += action
        self.record_observation()
        self.actions.append(action)
        # print(np.sum(self.board), self.total + 1)
        # assert np.sum(self.board) == self.total + 1
        # self.total += 1
        return self.immediate_return

    def action(self, action_index: int) -> np.array:
        z = np.zeros(9, dtype=np.int)
        z[action_index] = 1
        return z.reshape((3, 3))

    @property
    def immediate_return(self) -> int:
        if self.win_x:
            self.end = True
            return 1
        if self.win_o:
            self.end = True
            return -1
        elif self.draw:
            self.end = True
            return 0
        else:
            return 0

    @property
    def mask(self):
        return np.isclose(np.sum(self.board, axis=0), 0).astype(int).reshape(-1)

    def valid_action(self, action):
        x, y = np.where(action == 1.0)
        return np.isclose(np.sum(self.board[:, x, y]), 0)

    @property
    def draw(self):
        return np.allclose(self.board[0] + self.board[1], 1)

    @property
    def win_x(self):
        board = self.board[0] - self.board[1]
        vertical = np.isclose(np.sum(board, axis=0), 3).any()
        horizontal = np.isclose(np.sum(board, axis=1), 3).any()
        diagonals = np.isclose([np.trace(board), np.trace(np.rot90(board))], 3).any()
        return vertical or horizontal or diagonals

    @property
    def win_o(self):
        board = self.board[0] - self.board[1]
        vertical = np.isclose(np.sum(board, axis=0), -3).any()
        horizontal = np.isclose(np.sum(board, axis=1), -3).any()
        diagonals = np.isclose([np.trace(board), np.trace(np.rot90(board))], -3).any()
        return vertical or horizontal or diagonals

    @property
    def show(self):
        squares = [" ", " ", " ", " ", " ", " ", " ", " ", " "]
        for k in range(9):
            if self.board[0].reshape(-1)[k] == 1: squares[k] = "x"
            if self.board[1].reshape(-1)[k] == 1: squares[k] = "o"
        print(" ___\n|"+"".join(squares[:3]) + "|\n|" + "".join(squares[3:6]) + "|\n|" + "".join(squares[6:])+"|\n ---")
