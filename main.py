import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=280)

from tqdm import trange
import torch
from muzero import Muzero
from environments import TicTacToe
from mcts import Tree
from network import Network
import random
import sys

my_seed = 1954
np.random.seed(my_seed)

#initialising from command line
nb_epochs = int(sys.argv[1]) #15+
#print("Epochs = ", str(nb_epochs))
nb_episodes = int(sys.argv[2]) #1000+
#print("Episodes = ", str(nb_episodes))
nb_opti_steps = int(sys.argv[3]) #500+
#print("Opti_steps = ", str(nb_opti_steps))
nb_leaves_per_move = int(sys.argv[4]) #30+
#print("Leaves = ", str(nb_leaves_per_move))
nb_games = int(sys.argv[5]) #100
#print("Games = ", str(nb_games))


my_seed = int(sys.argv[6])

variables = [nb_epochs, nb_episodes, nb_opti_steps, nb_leaves_per_move, nb_games, my_seed]
for v in variables:
    print(v, end=' ')


################################################
############## DRIVER PROGRAM ##################
################################################

mu_agent = Muzero(TicTacToe) #creating the agent
random_agent = Muzero(TicTacToe) #creating the random adversary

game = TicTacToe()

# traininig model
for _ in range(nb_epochs):
    mu_agent.REPLAY_BUFFER = [] #init agent's replay buffer RB

    for _ in trange(nb_episodes):  # playing n plays from beginning to end
        with torch.no_grad(): #what happens when too many games?
            mu_agent.REPLAY_BUFFER.append(mu_agent.mcts.full_episode(nb_leaves_per_move))

    for _ in trange(nb_opti_steps):  # optimisation
        mu_agent.optimize_step()
    mu_agent.network.save('./models/new_with_script.ptc')

