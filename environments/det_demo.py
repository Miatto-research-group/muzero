from gate_synthesis import *
from operators import *
import numpy as np
from visu import *

np.random.seed(1954)

q1_gates = [X, Y, Z, S, H, T]
q2_gates = [CNOT]
init = np.tensordot(I, I, axes=0)

game = GateSynthesis(SWAP, init, 500, q1_gates, q2_gates)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
print(game.curr_unitary)

a = (CNOT, (0, 1))
print(f"Selected action : {a}")
rwd0 = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd0}")
print(game.curr_unitary)

print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
b = (CNOT, (1, 0))
print(f"Selected action : {b}")
rwd1 = game.step(b)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd1}")
print(game.curr_unitary)


print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
c = (CNOT, (1, 0))
print(f"Selected action : {c}")
rwd2 = game.step(c)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd2}")
print(game.curr_unitary)

print(f"End result current == target? {np.allclose(game.curr_unitary, game.target_unitary)}")