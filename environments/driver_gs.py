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
a = (CNOT, (0, 1))
print(f"Selected action : {a}")
rwd0 = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd0}")


print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
b = (CNOT, (1, 0))
print(f"Selected action : {b}")
rwd1 = game.step(b)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd1}")

print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}")
c = (CNOT, (1, 0))
print(f"Selected action : {c}")
rwd2 = game.step(c)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd2}")

print(f"End result current == target? {np.allclose(self.curr_unitary, self.target_unitary)}")


print("######################################################")
print("Now, random game!")
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

#game.play_one_episode(10)
"""
for i in range(50):
    b = game.select_explicit_action(5)
    print(f"Selected action : {b}")
    rwd1 = game.step(b)
    print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, received reward {rwd1}")
"""
print("####################################")





