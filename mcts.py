import numpy as np

c1 = 1.25
c2 = 19652
gamma = 0.997


class MCTS:
    def __init__(self, initial_state: list, prediction: callable, dynamics: callable, MDP: bool = False):
        self.reset()
        self.MDP = MDP
        self.prediction = prediction
        self.dynamics = dynamics
        policy, _ = self.prediction(initial_state)
        self.add_node(initial_state, policy)
        # TODO: self.mask for root node

    def reset(self):
        self.head = 0
        self.N = []  # visits
        self.P = []  # policies
        self.Q = []  # values
        self.R = []  # rewards
        self.S = []  # state_transitions (indices of self.states)
        self.states = []  # states
        self.trajectory = []
        self.prediction = None
        self.dynamics = None


    def add_node(self, state, policy):
        A = len(policy)
        self.N.append(np.zeros_like(policy))
        self.P.append(policy)
        self.Q.append(np.zeros_like(policy))
        self.R.append(np.zeros_like(policy))
        self.S.append([None for _ in range(A)])
        self.states.append(state)

    @property
    def policy_weights(self):
        N = np.array(self.N[self.head])
        return (np.sqrt(np.sum(N)) / (1 + N)) * (c1 + np.log(np.sum(N) + c2 + 1) - np.log(c2))

    def selection(self):
        Qmin = np.min(self.Q)
        Qmax = np.max(self.Q)
        if Qmax > Qmin:
            Qnorm = (self.Q[self.head] - Qmin)/(Qmax - Qmin)
        else:
            Qnorm = self.Q[self.head]
        b = Qnorm + self.P[self.head] * self.policy_weights
        action = np.random.choice(np.flatnonzero(b == b.max()))  # argmax with random tie-break
        if self.S[self.head][action] is None:
            self.expansion(action)
        else:
            self.trajectory.append((self.head, action))
            self.head = self.S[self.head][action]

    def expansion(self, action):
        state, reward = self.dynamics(self.states[self.head])
        policy, value = self.prediction(self.states[self.head], self.int_to_action(action))
        self.add_node(state, np.squeeze(policy))
        self.R[self.head][action] = np.squeeze(reward) if self.MDP else 0 # immediate reward not 0 for MDP
        self.S[self.head][action] = len(self.states) - 1
        self.trajectory.append((self.head, action))
        self.head = self.S[self.head][action]
        self.backup(np.squeeze(value))

    def backup(self, value):
        R = [self.R[node][action] for (node, action) in self.trajectory]
        for k, (node, action) in enumerate(reversed(self.trajectory)):
            G = sum(gamma ** j * R[j - k] for j in range(k)) + gamma ** k * value
            self.Q[node][action] = (self.N[node][action] * self.Q[node][action] + G) / (self.N[node][action] + 1)
            self.N[node][action] += 1
        self.head = 0
        self.trajectory = []

    def search(self, num_simulations: int, Temperature: float = 1.0):
        'Returns the policy and value found after a given number of simulations'
        for _ in range(num_simulations):
            self.selection()

        root_N = self.N[0]
        policy = root_N ** (1 / Temperature) / np.sum(root_N ** (1 / Temperature))
        return policy, self.Q[0] @ policy  # NOTE not sure this is how nu should be computed

    @staticmethod
    def int_to_action(action_index: int):
        NotImplementedError()
