import numpy as np

class TicTacToe:
    num_observations = 1 # number of last game states to give to the MCTS
    
    def __init__(self):
        self.board = np.zeros((3,3), dtype=np.float32)
        self.turn = 1
        self._observations = [self.board.reshape(-1).copy()]
        self.end = False

    def play(self, action:int) -> int:
        x = action//3
        y = action%3
        self.board[x,y] = self.turn # ±1
        self.turn *= -1
        self._observations.append(self.board.reshape(-1).copy())
        return self.immediate_return

    @property
    def immediate_return(self):
        vertical = np.sum(self.board, axis = 0)
        horizontal = np.sum(self.board, axis = 1)
        diagonals = np.array([np.trace(self.board), np.trace(np.rot90(self.board))])
        win_x = np.isclose(vertical, 3).any() or np.isclose(horizontal, 3).any() or np.isclose(diagonals, 3).any()
        win_o = np.isclose(vertical, -3).any() or np.isclose(horizontal, -3).any() or np.isclose(diagonals, -3).any()
        if win_x:
            self.end = True
            return 1
        elif win_o:
            self.end = True
            return -1
        else:
            if self.board_full:
                self.end = True
            return 0

    @property
    def observations(self):
        return self._observations[-self.num_observations:] 

    @property
    def legal_moves_mask(self):
        return np.isclose(self.board.reshape(-1), 0)

    @property
    def board_full(self):
        return np.isclose(np.sum(np.abs(self.board)), 9)

    