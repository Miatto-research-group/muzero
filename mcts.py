from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from environments import TicTacToe as Environment
from environments import Game

class Node:
    def __init__(self, parent, policy, state):
        self.parent = parent # (node, action) pair
        self.children = defaultdict(lambda: None)
        self.N = np.zeros(len(policy), dtype=int)
        self.P = np.array(policy)
        self.Q = np.zeros(len(policy), dtype=float)
        self.state = state

    def __repr__(self):
        return f"root visits: {self.N}"

@dataclass(repr=False)
class Episode:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    policies: list = field(default_factory=list)

    def store_data(self, oarvp:tuple):
        o,a,r,v,p = oarvp
        self.observations.append(np.array(o))
        self.actions.append(np.array(a))
        self.rewards.append(r)
        self.values.append(v)
        self.policies.append(p)
    
    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        env = Environment()
        for action in self.actions:
            env.play(action)
            env.show
        return ''

class Tree:
    def __init__(self, network, environment, observations = None):
        self.network = network
        self.environment = environment
        if observations is None:
            observations = self.environment.state[None,...]
        root_state = self.network.representation(observations)
        policy, _ = self.network.prediction(root_state)
        self.root = Node(parent=None, policy=policy, state=root_state)

    def reset(self):
        self.environment.__init__()
        self.__init__(self.network, self.environment)

    def select(self, node: Node, mask=None) -> int:
        "Returns an action to take from a node"
        pUCT = node.Q + node.P*np.sqrt(np.sum(node.N))/(node.N + 1)
        pUCT -= np.min(pUCT)
        if mask is not None:
            pUCT *= mask
            if np.allclose(pUCT, 0):
                return np.random.choice(np.flatnonzero(mask))
        return np.argmax(pUCT)

    def expand(self, node: Node, action: int) -> tuple:
        "Returns a new node and value of the selected action"
        state, _ = self.network.dynamics(node.state, self.environment.action(action)) #TODO: this isn't pretty
        policy, value = self.network.prediction(state)
        # print(state, value)
        new_node = Node(parent = (node, action) , policy = policy, state = state)
        node.children[action] = new_node
        return new_node, value

    def backup(self, node: Node, value: float):
        "Backs up the information about a newly added node"
        while node.parent is not None:
            node, action = node.parent
            node.Q[action] = (node.N[action]*node.Q[action] + value)/(node.N[action] + 1)
            node.N[action] += 1
        
    def add_leaf_and_backup(self, mask:np.array):
        "Keeps exploring until a new leaf is added to the tree (and backup)"
        node = self.root
        action = self.select(node, mask) # masking only the 1st action from the root
        while node.children[action] is not None:
            node = node.children[action]
            action = self.select(node)
        leaf, value = self.expand(node, action)
        self.backup(leaf, value)

    def policy(self, num_simulations: int, mask: np.array) -> np.array:
        "Performs a number of explorations and returns a policy for the action from the root node"
        for k in range(num_simulations):
            self.add_leaf_and_backup(mask)
        return self.root.N/np.sum(self.root.N)
        
    def full_episode(self, leaves_per_move:int = 50):
        "plays out a full episode, searching on a fixed number of leaves per move"
        self.reset()
        episode = Episode()
        while not self.environment.end:
            pi = self.policy(leaves_per_move, mask=self.environment.mask)
            action = np.random.choice(len(pi), p = pi) #choosing according to proba distribution of policy
            self.environment.play(action)
            episode.store_data((self.environment.state, action, self.environment.reward, self.root.Q[action], pi))
            self.root = self.root.children[action]
            self.root.N *= self.environment.mask # set visits to 0 for upcoming illegal moves
        if issubclass(self.environment.__class__, Game): #if it's a game and not just an MDP with single player
            u = self.environment.reward # final reward
            winner = (len(episode.values) - 1)%self.environment.num_players
            episode.values = [u if k%self.environment.num_players == winner else -u for k in range(len(episode.values))]
        return episode

    def move(self, env_state, leaves_per_move:int):
        """

        :param env_state:
        :param leaves_per_move:
        :return: an action selected at random according to proba distribution pi
        """
        self.environment.from_state(env_state)
        root_state = self.network.representation(self.environment.state)
        policy, _ = self.network.prediction(root_state)
        self.root = Node(parent=None, policy=policy, state=root_state)
        # print(self.environment.mask)
        pi = self.policy(leaves_per_move, mask=self.environment.mask)
        return np.random.choice(len(pi), p = pi)
