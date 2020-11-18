import numpy as np
import pytest
from environments import TicTacToe

T = TicTacToe()

def test_initial_board_is_Empty():
    T.__init__()
    assert np.allclose(T.board, 0)

def test_after_move_board_is_notEmpty():
    T.__init__()
    T.play(1)
    assert T.board[0,0,1] == 1

def test_turn():
    T.__init__()
    assert T.turn == 0
    T.play(1)
    assert T.turn == 1
    T.play(0)
    assert T.turn == 0

def test_initial_observations_all_zeros():
    T.__init__()
    assert np.allclose(T.observations, 0)

def test_new_observations_are_last_board_position():
    T.__init__()
    T.play(1)
    assert np.allclose(T.observations[-1], T.board)

def test_init_from_observations():
    T1 = TicTacToe()
    T1.play(1)
    T.from_observations(T1.observations)
    T.play(2)
    T1.play(2)
    assert np.allclose(T.observations, T1.observations)

def test_play():
    T.__init__()
    T.play(1)
    assert T.board[0,0,1] == 1
    T.play(2)
    assert T.board[1,0,2] == 1

def test_win_x():
    T.__init__()
    T.play(0)
    T.play(3)
    T.play(1)
    T.play(4)
    T.play(2)
    assert T.win_x == True
    assert T.win_o == False
    assert T.draw == False

def test_win_o():
    T.__init__()
    T.play(0)
    T.play(3)
    T.play(1)
    T.play(4)
    T.play(6)
    T.play(5)
    assert T.win_x == False
    assert T.win_o == True
    assert T.draw == False
