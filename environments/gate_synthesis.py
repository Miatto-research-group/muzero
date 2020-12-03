import numpy as np
from collections import deque
from envs import Game
from operators import *
import itertools

np.random.seed(1954)

class GateSynthesis(Game):
    num_players = 1
    num_gates = 7
    num_qbits = 3
    num_observations = 3 # number of states to pass to the representation network

    def __init__(self, target_unitary:np.array, curr_unitary:np.array, max_rwd: int, q1_gates:[np.array], q2_gates:[np.array] ):
        super().__init__()
        #self.state = np.zeros((1, 3, 3), dtype=np.int) #???
        self.curr_unitary = curr_unitary
        self.prev_unitary = curr_unitary
        self.target_unitary = target_unitary
        self.max_rwd = max_rwd
        self.q1_gates = q1_gates
        self.q2_gates = q2_gates
        self.nb_steps = 0

    @property
    def turn(self): #no need for one player
        return 1 #always player 1's turn


    def dist_to_target(self, unitary):
        """
        Computes distance between two unitary operators
        Natively dist between curr_unitary and target,
        but possibly between some unitary and target
        """
        return np.power(np.linalg.norm(self.target_unitary - unitary), 2)


    def qbit_num_to_tensor_index(self, n: int):
        m = (n + 1) * 2
        evens = list(filter(lambda x: x % 2 == 0, list(range(m))))
        return evens[(n - 1)]

    def apply_1q_gate(self, gate:np.array, qbit:int):
        qb_idx = self.qbit_num_to_tensor_index(qbit)
        self.curr_unitary = np.tensordot(gate, self.curr_unitary, axes=(qb_idx, 1))

    def apply_2q_gate(self, gate:np.array, qbitA:int, qbitB:int):
        qbA_idx = self.qbit_num_to_tensor_index(qbitA)
        qbB_idx = self.qbit_num_to_tensor_index(qbitB)
        self.curr_unitary = np.tensordot(gate, self.curr_unitary, axes=([qbA_idx,qbB_idx],[1,3]))

    def step(self, action):
        """
        Takes a step into the Hilbert space, applying a matrix
        """
        (gate, qbit) = action
        if (gate.shape == (2, 2, 2, 2)):  # 2qb
            (qbA, qbB) = qbit
            self.apply_2q_gate(gate, qbA, qbB)  # 1qb
        elif (gate.shape == (2, 2)):
            self.apply_1q_gate(gate, qbit)
        else:
            raise ValueError('Unsupported gate dimension')
        self.nb_steps += 1
        return self.reward

    @property
    def actions(self):
        """
        Returns all possible combination of actions available to the agent at time T
        """
        num_qbits = np.int(len(self.curr_unitary) / 2)
        q1_actions = list(itertools.product(self.q1_gates, range(num_qbits)))
        all_2q_permutations = list(itertools.product(range(num_qbits), range(num_qbits)))
        coherent_2q_permutations = b = list(filter(lambda x: x[0] != x[1], all_2q_permutations))
        q2_actions = list(itertools.product(self.q2_gates, coherent_2q_permutations))
        return q1_actions + q2_actions

    def select_random_action(self):
        poss_actions = self.actions
        idx = np.random.randint(0,len(poss_actions))
        return poss_actions[idx]

    def select_explicit_action(self, n:int):
        poss_actions = self.actions
        return poss_actions[n]

    @property
    def reward(self) -> int:
        """
        The reward is as follows
        max_rwd upon reaching final target unitary
        + step_rwd which is the improvement in distance from last position to new one
        So if the agent is closer, then step_rwd is positive,
        if agent is further away, step rwd is negative
        """
        target_rwd = int(np.allclose(self.target_unitary, self.curr_unitary)) * self.max_rwd
        step_rwd = self.dist_to_target(self.prev_unitary) - self.dist_to_target(self.curr_unitary)
        return target_rwd + step_rwd


    @property
    def end(self):
        return self.has_won or self.is_stuck

    @property
    def has_won(self):
        return np.allclose(self.curr_unitary, self.target_unitary)

    @property
    def is_stuck(self):  # TODO decide when to consider that this exploration has failed and agent is stuck
        return False

    def play_one_episode(self, th:int):
        rwd = 0
        while ((not self.has_won) and (self.nb_steps < th)):
            act = self.select_random_action()
            rwd += self.step(act)
            print(f"ep with rwd {rwd}")
        if (self.has_won):
            print(f"Agent found unitary in {self.nb_steps} with final reward {rwd}")
        else:
            print(f"Agent did not find unitary in {self.nb_steps}")
        return rwd

################################################################"
    """
    def play(self, action) -> int:  # drastic change
        if type(action) is int:
            action = self.action(action)
        if not self.valid_action(action):
            self.show
            raise ValueError(f"invalid action \n{action}")
        self.state[self.turn] += action
        # print("Play ", self.reward, "\n",  self.state[self.turn], flush=True)
        return self.reward

    
    
    @property
    def mask(self):
        return 1 - (self.state[0] + self.state[1]).reshape(-1)

    def valid_action(self, action):
        return action.reshape(-1) @ self.mask > 0

    @property
    def show(self):
        pass

"""
######################################################"
    def apply_1q_gate_to_q1(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axis=(0,1))

    def apply_1q_gate_to_q2(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axis=(2,1))

    def apply_1q_gate_to_q3(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axis=(4,1))

    def apply_2q_gate_to_q12(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axes=([0,2],[1,3]))

    def apply_2q_gate_to_q23(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axes=([2,4],[1,3]))

    def apply_2q_gate_to_q13(gate:np.array):
        return np.tensordot(gate, self.curr_unitary, axes=([0,4],[1,3]))

#############################################################"

