import numpy as np
c1 = 1.25
c2 = 19652
gamma = 0.997


class Tree:
    def __init__(self):#, observations: list, model: tuple):
        self.reset()

    def reset(self):
        self.head = 0
        self.N = [] # visits
        self.P = [] # policies
        self.Q = [] # values
        self.R = [] # rewards
        self.S = [] # state_transitions (indices of self.states)
        self.states = [] # states
        self.trajectory = []
        self.f = None
        self.g = None
        self.h = None
    
    def set_obs_model(self, observations: list, model: tuple):
        self.f, self.g, self.h = model
        self.add_node(self.h(observations))

    def add_node(self, s):
        policy, value = self.f(s)
        A = policy.shape[1] # number of actions from s
        self.N.append([0 for _ in range(A)])
        self.P.append(np.squeeze(policy))                       # NOTE: not sure it's this. See page 12 (Expansion) in https://arxiv.org/abs/1911.0826
        self.Q.append([np.squeeze(value) for _ in range(A)])    # NOTE: not sure it's this. See page 12 (Expansion) in https://arxiv.org/abs/1911.08265
        self.R.append([0 for _ in range(A)])
        self.S.append([None for _ in range(A)])
        self.states.append(s.numpy())

    @property
    def P_weights(self):
        N = np.array(self.N[self.head])
        return np.sqrt(np.sum(N))/(1 + N)*(c1 + np.log(np.sum(N) + c2 + 1) - np.log(c2))

    def selection(self):
        b = np.array(self.Q[self.head]) + np.array(self.P[self.head]) * self.P_weights
        a = np.random.choice(np.flatnonzero(b == b.max())) # argmax with random tie-break

        if self.S[self.head][a] is None:
            self.expansion(a)
        else:
            self.trajectory.append((self.head, a))
            self.head = self.S[self.head][a]

    def expansion(self, a):
        r, s = self.g(self.states[self.head], a)
        self.add_node(s)

        self.R[self.head][a] = np.squeeze(r)
        self.S[self.head][a] = len(self.states)-1
        
        self.trajectory.append((self.head, a))
        self.head = self.S[self.head][a]
        self.backup()
            
    def backup(self):
        v = self.Q[self.head][0] # Q[s][a] are all the same for all a for a new node
        R = [self.R[node][a] for (node, a) in self.trajectory]
        for k, (node, a) in enumerate(reversed(self.trajectory)):
            G = sum(gamma**j * R[j-k] for j in range(k)) + gamma**k * v
            self.Q[node][a] = (self.N[node][a]*self.Q[node][a]+G)/(self.N[node][a] + 1)
            self.N[node][a] += 1 
        self.head = 0
        self.trajectory = []
        
    def policy_value(self, num_simulations:int, T:float = 1.0):
        for _ in range(num_simulations):
            self.selection()

        root_N = np.array(self.N[0])
        pi = root_N**(1/T)/np.sum(root_N**(1/T))
        return pi, np.array(self.Q[0])@pi #NOTE not sure this is nu

    def __repr__(self):
        return str(self.N)