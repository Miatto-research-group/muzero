from gate_synthesis import *
from operators import *
import numpy as np
from visu import *

np.random.seed(1954)


q1_gates = [X, Y, Z, S, H, T, Tdag]
q2_gates = [CNOT]
init = np.tensordot(I, II, axes=0)

game = GateSynthesis(FREDKIN, init, 500, q1_gates, q2_gates)

print("##### DETERMINISTIC DRIVER DEMO FOR FREDKIN GATE #####")
print(f"Initial distance to target = {game.dist_to_target(game.curr_unitary)}")
print("- - - - - - - - - ")

print("Do step #0 : apply CNOT to q2, q1")
a = (CNOT, (2,1))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")


print("Do step #1 : apply H to q2")
a = (H, 2)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #2a : apply T to q0")
a = (T, 0 )
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #2b : apply T to q1")
a = (T, 1 )
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #2c : apply T to q2")
a = (T, 2)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #3 : apply CNOT to q1, q0")
a = (CNOT, (1,0))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #4 : apply CNOT to q2, q1")
a = (CNOT, (2,1))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #5 : apply CNOT to q0, q2")
a = (CNOT, (0,2))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #6 : apply Tdag to q1")
a = (Tdag, 1)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #7 : apply T to q2")
a = (T, 2)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #8 : apply CNOT to q0, q1")
a = (CNOT, (0,1))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #9 : apply Tdag to q0")
a = (Tdag, 0)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #10 : apply Tdag to q1")
a = (Tdag, 1)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #11 : apply CNOT to q2, q1")
a = (CNOT, (2,1))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #12 : apply CNOT to q0, q2")
a = (CNOT, (0,2))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #13 : apply CNOT to q1, q0")
a = (CNOT, (1,0))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #14 : apply H to q2")
a = (H, 2)
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print("Do step #15 : apply CNOT to q2, q1")
a = (CNOT, (2,1))
r = game.step(a)
print(f"Distance to target = {game.dist_to_target(game.curr_unitary)}, \t rwd {r}")
print("- - - - - - - - - ")

print(f"End result current == target? {np.allclose(game.curr_unitary, game.target_unitary, atol=0.5, rtol=0.5)}")
