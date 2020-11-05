import numpy as np
from environments import TicTacToe as Environment
from loguru import logger
# import sys

# logger.add("detailed.log", filter=lambda record: record["extra"].get("name") == "detailed")
# logger.add(sys.stdout, filter=lambda record: record["extra"].get("name") == "console")
# logger_detailed = logger.bind(name="detailed")
# logger_console = logger.bind(name="console")

c1 = 1.25
c2 = 19652
gamma = 1 #TODO discount in games is 1

class MCTS:
    def __init__(self, network):
        self.environment = Environment()
        self.network = network
        self.reset(self.environment.observations)

    def reset(self, observations):
        self.head = 0
        self.N = []  # visits
        self.P = []  # policies
        self.Q = []  # values
        self.R = []  # rewards
        self.S = []  # state_transitions (indices of self.states)
        self.states = []
        self.trajectory = []
        self.Qmax = 0
        self.Qmin = 0
        initial_state = self.network.representation(observations)
        policy, _ = self.network.prediction(initial_state)
        self.add_node(initial_state, policy)

    def add_node(self, state, policy):
        self.states.append(state)
        self.N.append(np.zeros_like(policy))
        self.P.append(policy.numpy())
        # logger.info(f"new-node policy: {self.P[-1]}")
        self.Q.append(np.zeros_like(policy))
        self.R.append(np.zeros_like(policy))
        self.S.append([None for p in policy])

    @property
    def policy_weights(self):
        N = np.array(self.N[self.head])
        # NOTE: this includes + 0.1 in the sqrt with respect to eq. (2) at page 12 of https://arxiv.org/abs/1911.08265
        return (np.sqrt(np.sum(N)) / (1 + N)) * (c1 + np.log(np.sum(N) + c2 + 1) - np.log(c2))

    def select_action(self):
        Qnorm = (self.Q[self.head] - self.Qmin) / (self.Qmax - self.Qmin) if self.Qmax > self.Qmin else self.Q[self.head]
        b = Qnorm + (self.P[self.head]) * self.policy_weights #NOTE we added 1e-6, but it's not in the paper
        if self.head == 0:
            b = b * self.environment.mask.reshape(-1)
        logger.info(f"{b}, {np.flatnonzero(b == b.max())}")
        action = np.random.choice(np.flatnonzero(b == b.max())) # argmax with random tie-break
        self.trajectory.append((self.head, action))
        return action

    def expand(self, action: int):
        state, reward = self.network.dynamics(self.states[self.head], self.environment.action(action)) # TODO: pass action with batch dimension?
        policy, value = self.network.prediction(self.states[self.head])
        self.add_node(state, policy)
        self.R[self.head][action] = reward if self.environment.MDP else 0  # immediate reward 0 for games
        self.S[self.head][action] = len(self.states) - 1
        return value
        
    def backup(self, value):
        R = [self.R[node][action] for (node, action) in self.trajectory]
        for k, (node, action) in enumerate(reversed(self.trajectory)):
            G = sum(gamma ** j * R[j - k] for j in range(k)) + gamma ** k * value
            self.Q[node][action] = (self.N[node][action] * self.Q[node][action] + G) / (self.N[node][action] + 1)
            if (val := np.max(self.Q[node][action])) > self.Qmax:
                self.Qmax = val
            if (val := np.min(self.Q[node][action])) < self.Qmin:
                self.Qmin = val
            self.N[node][action] += 1
        self.head = 0
        self.trajectory = []


    def is_new_node(self, action):
        return self.S[self.head][action] is None


    def search(self, num_simulations: int = 35, Temperature: float = 1.0):
        "Returns the policy and value (for a single step) found after a given number of simulations"
        sim = 0
        while sim < num_simulations:
            action = self.select_action()
            if self.is_new_node(action):
                logger.info(f"sim: {sim} | trajectory: {self.trajectory}")
                self.backup(self.expand(action))
                sim += 1
            else:
                self.head = self.S[self.head][action]
        root_N = self.N[0]
        policy = root_N ** (1 / Temperature) / np.sum(root_N ** (1 / Temperature))
        return policy, self.Q[0] @ policy  # NOTE not sure this is how nu should be computed


    def self_play(self, num_simulations:int = 35):
        "plays a full episode"
        self.environment.__init__()
        policies = []
        rewards = []
        observations = [self.environment.observations.copy()]
        while not self.environment.end:
            self.reset(self.environment.observations)
            policy, _ = self.search(num_simulations)
            policies.append(policy)
            action = np.random.choice(len(policy), p=policy / np.sum(policy))
            logger.info(f"action {action} from policy: {policy}")
            rewards.append(self.environment.play(action))
            observations.append(self.environment.observations.copy())
        return observations, self.environment.actions, policies, rewards


    def rollout(self, observations: list, K: int = 5):
        "plays K steps from any given environment state (observations)"
        self.reset(observations)
        self.environment.from_observations(observations)
        policies = []
        values = []
        actions = []
        for _ in range(K):
            self.reset(self.environment.observations)
            policy, value = self.search()
            policies.append(policy)
            values.append(value)
            action = np.random.choice(len(policy), p=policy / np.sum(policy))
            actions.append(action)
            if self.environment.end:
                for _ in range(K - len(actions)):
                    policies.append(policies[-1])
                    values.append(values[-1])
                break
            else:
                self.environment.play(action)
        return policies, values, actions