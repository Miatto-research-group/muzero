import numpy as np

np.set_printoptions(suppress=False, linewidth=280)
from itertools import chain
from tqdm import trange

from environments import TicTacToe as Environment
from mcts import MCTS
from networks import representation, dynamics, prediction

import torch


class Muzero:
    def __init__(self, Environment):
        self.representation = representation()
        self.dynamics = dynamics()
        self.prediction = prediction()
        self.environment = Environment()
        self.mcts = MCTS(self.dynamics, self.prediction, self.representation)
        self.REPLAY_BUFFER = []
        self.optimizer = torch.optim.Adam(chain(self.representation.parameters(), self.dynamics.parameters(), self.prediction.parameters()), lr=0.01)

    def muzero_rollout(self, observations: list, actions: list, K: int = 5):
        policies = []
        values = []
        state = self.representation(observations)
        for action in actions[:K]:
            state, reward = self.dynamics(state, action)
            policy, value = self.prediction(state)
            policies.append(policy.squeeze())
            values.append(value.squeeze())
        for k in range(K - len(actions)):
            policies.append(policies[-1])
            values.append(values[-1])
        return policies, values

    def loss(self):
        g = np.random.choice(len(self.REPLAY_BUFFER))
        observations, actions, policies, observed_reward = self.REPLAY_BUFFER[g]
        move = np.random.choice(len(actions))
        print(f"chose game {g} and move {move}")
        mcts_policies, _, _ = self.mcts_rollout(observations[move], K=5)
        muzero_policies, muzero_values = self.muzero_rollout(observations[move], actions[move:], K=5)
        loss_v = ((observed_reward - torch.tensor(muzero_values)) ** 2).sum()
        loss_p = 0
        for pi, p in zip(mcts_policies, muzero_policies):
            loss_p += torch.dot(torch.tensor(pi), torch.log(p + 1e-9))
        return loss_p + loss_v

    def optimize_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        output = self.loss()
        output.backward()
        self.optimizer.step()
