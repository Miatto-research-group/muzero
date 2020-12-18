from gate_synthesis import *
from operators import *
import numpy as np
from visu import *

np.random.seed(1954)

q1_gates = [X, Y, Z, S, H, T]
q2_gates = [CNOT]
init = np.tensordot(I, I, axes=0)

revCNOT = np.array([[[[1, 0],[0, 0]], [[0 ,0],[0 ,1]]], [[[0 ,0],[0, 1]],[[1, 0],[0, 0]]]])
beta = np.array([[[[1, 0],[0, 0]], [[0 ,0],[1 ,0]]], [[[0 ,0],[0, 1]],[[0, 1],[0, 0]]]])

game = GateSynthesis(SWAP, init, 500, q1_gates, q2_gates)
#print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")


print(f"Initial distance to target = {game.dist_to_target(game.curr_unitary)}")
print("- - - - - - - - - ")

a = (CNOT, (0, 1))
rwd0 = game.step(a)
print("Do step #1")
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd0}")
print("Step correct?", np.allclose(game.curr_unitary, CNOT))

print("- - - - - - - - - ")

b = (CNOT, (1, 0))
rwd1 = game.step(b)
print("Do step #2")
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd1}")
print("Step correct?", np.allclose(game.curr_unitary, beta))


print("- - - - - - - - - ")

c = (CNOT, (0, 1))
rwd2 = game.step(c)
print("Do step #3")
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd2}")
print("Step correct?", np.allclose(game.curr_unitary, SWAP))

print(f"End result current == target? {np.allclose(game.curr_unitary, game.target_unitary)}")
