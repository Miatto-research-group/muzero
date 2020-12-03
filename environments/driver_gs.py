from gate_synthesis import *
from operators import *
import numpy as np

np.random.seed(1954)

q1_gates = [X, Y, Z, S, H, T]
q2_gates = [CNOT]
init = np.tensordot(I, I, axes=0)
target = SWAP
total_rwd = 0



game = GateSynthesis(SWAP, init, 500, q1_gates, q2_gates)

print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
a = game.select_random_action()
print(f"Selected action : {a}")

rwd0 = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd0}")

b = game.select_explicit_action(5)
print(f"Selected action : {b}")
rwd1 = game.step(b)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd1}")

game.play_one_episode(1000)

print("####################################")





