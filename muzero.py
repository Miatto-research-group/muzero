import torch
from tqdm import trange
import numpy as np
np.set_printoptions(suppress=False, linewidth=280)

from mcts import Tree as MCTS
from network import Network

class Muzero:
    def __init__(self, Environment):
        self.environment = Environment()
        self.network = Network(Environment.num_observations)
        with torch.no_grad():
            self.mcts = MCTS(self.network, Environment)
        self.REPLAY_BUFFER = []
        self.optimizer = torch.optim.SGD(self.network.parameters, lr=0.01, weight_decay=0.001)

    def rollout(self, observations: list, actions: list, K: int = 5):
        policies = []
        values = []
        state = self.network.representation(observations)
        for action in actions[:K]:
            state, reward = self.network.dynamics(state, self.environment.action(action))
            policy, value = self.network.prediction(state)
            policies.append(policy.squeeze())
            values.append(value.squeeze())
        for k in range(K - len(actions)):
            policies.append(policies[-1])
            values.append(values[-1])
        return policies, values

    def loss(self):
        g = np.random.choice(len(self.REPLAY_BUFFER))
        episode = self.REPLAY_BUFFER[g]
        move = np.random.choice(len(episode.observations)-2)
        # print(f"starting from move {move} of game {g}")
        with torch.no_grad():
            rollout_episode = self.mcts.partial_rollout(episode, move, K=3)
        policies, values = self.rollout(episode.observations[move], episode.actions[move:], K=3)

        loss_v = 0
        for v in values:
            loss_v += (rollout_episode.values[-1] - v) ** 2
        loss_p = 0
        for pi, p in zip(rollout_episode.policies, policies):
            loss_p += torch.dot(pi, torch.log(p + 1e-9))
        return loss_v + loss_p

    def optimize_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        output = self.loss()
        output.backward()
        self.optimizer.step()

    def fill_replay_buffer(self, games:int):
        with torch.no_grad():
            self.mcts = MCTS(self.network)
        self.REPLAY_BUFFER = []
        with torch.no_grad():
            for g in trange(games):
                self.REPLAY_BUFFER.append(self.mcts.self_play())
