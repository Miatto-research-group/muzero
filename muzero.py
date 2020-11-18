import torch
from tqdm import trange
import numpy as np
np.set_printoptions(suppress=False, linewidth=280)

from mcts import Tree as MCTS
from mcts import Episode
from network import Network

class Muzero:
    def __init__(self, Environment):
        self.environment = Environment()
        self.network = Network(Environment.num_observations)
        with torch.no_grad():
            self.mcts = MCTS(self.network, Environment())
        self.REPLAY_BUFFER = []
        self.optimizer = torch.optim.Adam(self.network.parameters, lr=0.001, weight_decay=0.001)

    def rollout(self, observation, actions: list, K: int = 5):
        "plays K hypothetical steps from any given observation"
        episode = Episode()
        state = self.network.representation(observation)
        for action in actions[:K]:
            state, reward = self.network.dynamics(state, self.environment.action(action))
            policy, value = self.network.prediction(state)
            episode.store_data((None, None, reward, value, policy))
        for k in range(K - len(actions[:K])):
            episode.store_data((None, None, reward, value, policy))
        return episode

    def loss(self):
        g = np.random.choice(len(self.REPLAY_BUFFER))
        episode = self.REPLAY_BUFFER[g]
        move = np.random.choice(len(episode.actions)-2) # TODO: check the -2 (should make sure there's at least one move left)
        rollout_episode = self.rollout(episode.observations[max(0, move+1-self.environment.num_observations): move+1], episode.actions[move:], K=5)
        loss_v = 0
        for z, v in zip(episode.values, rollout_episode.values):
            loss_v += (z - v) ** 2
        loss_p = 0
        for pi, p in zip(episode.policies, rollout_episode.policies):
            loss_p += torch.dot(torch.tensor(pi, dtype=torch.float), torch.log(p + 1e-9))
        return loss_v + loss_p

    def optimize_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        output = self.loss()
        output.backward()
        self.optimizer.step()

    def fill_replay_buffer(self, games:int):
        self.REPLAY_BUFFER = []
        with torch.no_grad():
            for g in trange(games):
                self.REPLAY_BUFFER.append(self.mcts.full_episode())
