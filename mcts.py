from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from environments import TicTacToe as Environment

class Node:
    def __init__(self, parent, policy, state):
        self.parent = parent
        self.children = defaultdict(lambda: None)
        self.N = np.zeros(len(policy), dtype=int)
        self.P = policy
        self.Q = np.zeros(len(policy), dtype=float)
        self.state = state

    def __repr__(self):
        return f"root visits: {self.N}"

@dataclass(repr=False)
class Episode:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    values: list = field(default_factory=list)
    policies: list = field(default_factory=list)

    def __repr__(self):
        env = Environment()
        for action in self.actions:
            env.play(action)
            env.show
        return ''

class Tree:
    def __init__(self, network, Environment):
        self.network = network
        self.environment = Environment()
        root_state = self.network.representation(self.environment.observations)
        policy, _ = self.network.prediction(root_state)
        self.root = Node(None, policy, root_state)

    def select(self, node: Node, mask=None) -> int:
        pUCT = (node.Q + node.P.numpy()*np.sqrt(np.sum(node.N))/(node.N + 1))
        if mask is not None:
            pUCT *= mask
            if np.allclose(pUCT, 0):
                pUCT = mask
        return np.random.choice(np.flatnonzero(pUCT==pUCT.max())) # argmax with random tie-break

    def expand(self, node: Node, action: int) -> Node:
        state, _ = self.network.dynamics(node.state, self.environment.action(action))
        policy, value = self.network.prediction(state)
        new_node = Node(parent = (node, action) , policy = policy, state = state)
        node.children[action] = new_node
        return new_node, value

    def backup(self, node: Node, value: float):
        while node.parent is not None:
            node, action = node.parent
            node.Q[action] = (node.N[action]*node.Q[action] + value)/(node.N[action] + 1)
            node.N[action] += 1
        
    def explore(self, mask:np.array):
        node = self.root
        action = self.select(node, mask) # assumes env and tree are synced at the root
        while node.children[action] is not None:
            node = node.children[action]
            action = self.select(node)
        leaf, value = self.expand(node, action)
        self.backup(leaf, value)

    def search(self, num_simulations: int, mask: np.array) -> np.array:
        for k in range(num_simulations):
            self.explore(mask)
        return self.root.N/np.sum(self.root.N)
        
    def full_rollout(self, num_simulations:int = 50):
        self.environment.__init__()
        self.root.state = self.network.representation(self.environment.observations)
        episode = Episode()
        while not self.environment.end:
            pi = self.search(num_simulations, mask=self.environment.mask)
            action = np.random.choice(len(self.root.N), p = pi)
            self.environment.play(action)
            # self.environment.show
            episode.observations.append(self.environment.observations)
            episode.actions.append(action)
            episode.policies.append(self.root.P)
            episode.values.append(self.root.Q[action])
            self.root = self.root.children[action]
            self.root.N *= self.environment.mask
        return episode

    def partial_rollout(self, episode:Episode, move:int, K: int = 3, num_simulations:int = 100):
        "plays K steps from any given episode move"
        self.environment.from_observations(episode.observations[move])
        self.root.state = self.network.representation(self.environment.observations)
        rollout_episode = Episode()
        for _ in range(K):
            pi = self.search(num_simulations = num_simulations, mask=self.environment.mask)
            action = np.random.choice(len(self.root.N), p = pi) #TODO get_action()
            episode.actions.append(action)
            episode.policies.append(pi)
            episode.values.append(self.root.Q[action])
            if self.environment.end:
                for _ in range(K - len(episode.actions)):
                    policies.append(policies[-1])
                    values.append(values[-1])
                break
            else:
                self.environment.play(action)
                self.root = self.root.children[action]
                self.root.N *= self.environment.mask
        return episode
        
