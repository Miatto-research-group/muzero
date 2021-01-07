from gate_synthesis import *
from operators import *
import numpy as np
from visu import *

np.random.seed(1954)

q1_gates = [X, Y, Z, S, H, T]
q2_gates = [CNOT]
init = np.tensordot(I, I, axes=0)
target = SWAP
total_rwd = 0

game = GateSynthesis(SWAP, init, 500, q1_gates, q2_gates)

print(f"Initial distance to target = {game.dist_to_target(game.curr_unitary)}")
print("- - - - - - - - - ")
game = GateSynthesis(SWAP, init, 500, q1_gates, q2_gates)
game.play_one_episode(10000)

#plotting
nb_steps = list(range(10000))

y_tot_rwd = game.tot_reward_history
y_dist = game.distance_history
ys_rwd_dist = [y_tot_rwd, y_dist]
labs_rwd_dist = ["Rewards", "Distance to target"]
do_plot_2D(nb_steps, ys_rwd_dist, labs_rwd_dist, "perf_random_dist_rwd", "Performance of random agent", gnplt=False, x_lab="timestep", y_lab="perf")

y_pos = game.pos_reward_history
y_neg = game.neg_reward_history
ys = [y_pos, y_neg]
labs_pos_neg = ["Positive rewards", "Negative rewards"]
do_plot_2D(nb_steps, ys, labs_pos_neg, "perf_random_pos_neg", "Evolution of cumulated reward", logx=True, logy=False, gnplt=False, x_lab="timestep", y_lab="perf")
print("######################################################")




