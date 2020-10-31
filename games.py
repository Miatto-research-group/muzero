import numpy as np


class TicTacToe:
    num_observations = 1  # number of last game states to give to the MCTS

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self._observations = [self.board.copy()]
        self.turn = 1
        self.end = False

    def play(self, action: np.array) -> int: # assumes (legal) action is all zeros except new move
        self.board += action*self.turn
        self.turn *= -1
        self._observations.append(self.board.copy())
        return self.immediate_return

    @property
    def immediate_return(self):
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
    def observations(self):
        return self._observations[-self.num_observations :]

    @property
    def legal_moves_mask(self):
        return np.isclose(self.board, 0)

    @property
    def draw(self):
        return np.isclose(np.sum(np.abs(self.board)), 9) and np.isclose(np.sum(self.board), 0)

    @property
    def win_x(self):
        vertical   = np.isclose(np.sum(self.board, axis=0), 3).any()
        horizontal = np.isclose(np.sum(self.board, axis=1), 3).any()
        diagonals  = np.isclose([np.trace(self.board), np.trace(np.rot90(self.board)]), 3).any()
        return vertical or horizontal or diagonals

    @property
    def win_o(self):
        vertical   = np.isclose(np.sum(self.board, axis=0), -3).any()
        horizontal = np.isclose(np.sum(self.board, axis=1), -3).any()
        diagonals  = np.isclose([np.trace(self.board), np.trace(np.rot90(self.board)]), -3).any()
        return vertical or horizontal or diagonals
