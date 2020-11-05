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
    assert not np.allclose(T.board, 0)

def test_initially_observations_all_zeros():
    T.__init__()
    assert np.allclose(T.observations, 0)

def test_initially_observations_all_zeros():
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