import numpy as np
from collections import deque

class TicTacToe:
    num_observations = 3
    MDP = False

    def __init__(self):
        self._board = np.zeros((2, 3, 3), dtype=np.int)
        self.observations = deque(list(np.zeros((self.num_observations, 2, 3, 3))), maxlen=self.num_observations)
        self.end = False
        self.actions = []
        self.record_observation()

    def from_observations(self, observations):
        self._observations = observations
        turn = int(np.sum(observations[-1]) % 2)
        self._board = observations[-1][turn]
        self.end = self.win_o or self.win_x or self.draw

    @property
    def turn(self):
        return int(np.sum(self._board) % 2)

    def record_observation(self):
        self.observations.appendleft(self._board)

    def play(self, action) -> int:
        if type(action) in int:
            action = self.action(action)
        if not self.valid_action(self._board, action):
            raise ValueError(f"invalid action {action} with board {self._board}")
        self._board[self.turn] += action
        self.record_observation()
        self.actions.append(action)
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

    def mask(self, observation=None): # TODO: do we need observations?
        if observation is None:
            return np.isclose(self._board[self.turn], 0).astype(int)
        else:
            turn = int(np.sum(observation) % 2)
            return np.isclose(observation[turn], 0).astype(int)

    @staticmethod
    def valid_board(board):
        return np.sum(np.sum(board, axis=0) == 2) == 0

    @staticmethod
    def valid_action(board, action):
        x, y = np.where(action == 1.0)
        return np.isclose(np.sum(board[:, x, y]), 1)

    @property
    def draw(self):
        board = self._board[0] - self._board[1]
        return np.isclose(np.sum(np.abs(board)), 9) and np.isclose(np.sum(board), 0)

    @property
    def win_x(self):
        board = self._board[0] - self._board[1]
        vertical = np.isclose(np.sum(board, axis=0), 3).any()
        horizontal = np.isclose(np.sum(board, axis=1), 3).any()
        diagonals = np.isclose([np.trace(board), np.trace(np.rot90(board))], 3).any()
        return vertical or horizontal or diagonals

    @property
    def win_o(self):
        board = self._board[0] - self._board[1]
        vertical = np.isclose(np.sum(board, axis=0), -3).any()
        horizontal = np.isclose(np.sum(board, axis=1), -3).any()
        diagonals = np.isclose([np.trace(board), np.trace(np.rot90(board))], -3).any()
        return vertical or horizontal or diagonals

    @property
    def board(self):
        squares = ["", "", "", "", "", "", "", "", ""]
        for k in range(9):
            squares[k] = "x" if self._board[0].reshape(-1)[k] == 1 else " "
            squares[k] = "o" if self._board[1].reshape(-1)[k] == 1 else " "
        print("".join(squares[:3]) + "\n" + "".join(squares[3:6]) + "\n" + "".join(squares[6:]))
