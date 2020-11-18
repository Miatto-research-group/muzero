from network import Network
from mcts import Tree
from environments import TicTacToe
import random

net = Network(TicTacToe.num_observations)
net.load('models/tictactoe_best')
tree = Tree(net, TicTacToe)

if __name__ == "__main__":
    playing = input('press enter to play a game...')
    while playing=='':
        game = TicTacToe()
        tree.__init__(net, TicTacToe)
        
        if random.randint(0,1):
            ai_move = tree.search()
            game.play(ai_move)
            game.show

        while True:
            human_move = input("insert move [1-9]:")
            game.play(human_move-1)
            game.show
            if game.end:
                break

            ai_move = tree.search()
            game.play(ai_move)
            game.show
            if game.end:
                break

        input('press enter to play another game...')

