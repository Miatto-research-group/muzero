# TODO: import trained model, make it play against older version(s) and compute elo rating
import torch
import numpy as np
from tqdm import trange
from network import Network
from mcts import Tree
from environments import TicTacToe
import random

if __name__ == "__main__":

    net1 = Network(TicTacToe.num_observations)
    net1.load('models/ttt_candidate1.ptc')
    net2 = Network(TicTacToe.num_observations)
    net2.load('models/random.ptc')

    game = TicTacToe()
    with torch.no_grad():
        player = Tree(net1, game)
        random = Tree(net2, game)

    def play_a_game():
        while True:
            game.play(player.move(game.state, leaves_per_move=50))
            if game.end: break
            game.play(random.move(game.state, leaves_per_move=50))
            if game.end: break
        return game.reward

    score_white = 0
    with torch.no_grad():
        for g in (t:=trange(100)):
            score_white += play_a_game()
            t.set_description(f"Score as white: {score_white}")

    score_black = 0
    with torch.no_grad():
        for g in (t:=trange(100)):
            score_black += play_a_game()
            t.set_description(f"Score as black: {score_black}")
    
    
    print(f'Final Scores = W: {score_white} / B: {score_black}')