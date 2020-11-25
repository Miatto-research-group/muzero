# TODO: import trained model, make it play against older version(s) and compute elo rating
import torch
import numpy as np
from tqdm import trange
from network import Network
from mcts import Tree
from environments import TicTacToe
import random
np.set_printoptions(precision=6, suppress=True, linewidth=180)

np.random.seed(1954)

if __name__ == "__main__":

    net1 = Network(TicTacToe.num_observations)
    net1.load('models/ttt_candidate1.ptc')
    # net1.load('models/random.ptc')
    net2 = Network(TicTacToe.num_observations)
    net2.load('models/random.ptc')
    
    game = TicTacToe() #why twice?
    with torch.no_grad():
        player = Tree(net1, game)
        random = Tree(net2, game)

    def play_a_game(player1, player2):
        game = TicTacToe()
        player1.environment = game
        player2.environment = game
        while True:
            game.play(player1.move(game.state, leaves_per_move=20))
            if game.end:
                game.show
                return game.reward
            game.play(player2.move(game.state, leaves_per_move=20))
            if game.end:
                game.show
                return -game.reward
        

    score_white = 0
    with torch.no_grad():
        for g in (t:=trange(100)):
            score_white += play_a_game(player, random)
            t.set_description(f"Score as white: {score_white}")

    score_black = 0
    with torch.no_grad():
        for g in (t:=trange(100)):
            score_black -= play_a_game(random, player)
            t.set_description(f"Score as black: {score_black}")
    
    
    print(f'Final Scores = W: {score_white} / B: {score_black}')