import numpy as np

c1 = 1.25
c2 = 19652
gamma = 0.997


class Tree:
    def __init__(self):
        self.reset()

    def reset(self):
        self.head = 0
        self.N = []  # visits
        self.P = []  # policies
        self.Q = []  # values
        self.R = []  # rewards
        self.S = []  # state_transitions (indices of self.states)
        self.states = []  # states
        self.trajectory = []
        self.model = None

    def initialize(self, initial_state: list, model: callable):
        self.model = model
        state, policy, value, reward = self.model(initial_state)
        self.add_node(state, policy, value)

    def add_node(self, state, policy):
        A = len(policy)
        self.N.append([0 for _ in range(A)])
        self.P.append(policy)
        self.Q.append([0 for _ in range(A)])
        self.R.append([0 for _ in range(A)])
        self.S.append([None for _ in range(A)])
        self.states.append(state)

    @property
    def policy_weights(self):
        N = np.array(self.N[self.head])
        return np.sqrt(np.sum(N)) / (1 + N) * (c1 + np.log(np.sum(N) + c2 + 1) - np.log(c2))

    def selection(self):
        b = np.array(self.Q[self.head]) + np.array(self.P[self.head]) * self.policy_weights
        a = np.random.choice(np.flatnonzero(b == b.max()))  # argmax with random tie-break
        if self.S[self.head][a] is None:
            self.expansion(a)
        else:
            self.trajectory.append((self.head, a))
            self.head = self.S[self.head][a]

    def expansion(self, action):
        state, policy, value, reward = self.model(self.states[self.head])
        self.add_node(state.numpy(), np.squeeze(policy))
        self.R[self.head][action] = np.squeeze(reward)
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
        'Returns the policy and value found after num_simulations'
        for _ in range(num_simulations):
            self.selection()

        root_N = np.array(self.N[0])
        policy = root_N ** (1 / Temperature) / np.sum(root_N ** (1 / Temperature))
        return policy, np.array(self.Q[0]) @ policy  # NOTE not sure this is how nu should be computed

    def __repr__(self):
        return str(self.N)
