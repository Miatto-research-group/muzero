from operators import *
from random_gate_generator import *


gates1 = [X, Y, Z, S, H, T]
gates2 = [CNOT, SWAP]

#create 5 random gates of random complexity
targets = []
for _ in range(5):
    dice = np.random.randint(30)
    unitary, action_path = make_random_unitary(qbg1=gates1, qbg2=gates2, nb_steps=dice, size=3)
    targets.append((unitary, action_path))

for i in range(5):
    curr_u = III #reset
    (tu, ap) = targets[i]
    for a in ap:
        (g, qb) = a
        curr_u = apply_gate_on_qbits(a,curr_u)
    print("{}- Target==Current? : ".format(i), np.allclose(curr_u, tu))



