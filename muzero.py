import torch
from tqdm import trange
import numpy as np
np.set_printoptions(suppress=False, linewidth=280)

from mcts import MCTS
from network import Network

class Muzero:
    def __init__(self, Environment):
        self.environment = Environment()
        self.network = Network(Environment.num_observations)
        with torch.no_grad():
            self.mcts = MCTS(self.network)
        self.REPLAY_BUFFER = []
        self.optimizer = torch.optim.SGD(self.network.parameters, lr=0.01, weight_decay=0.001)

    def rollout(self, observations: list, actions: list, K: int = 5):
        policies = []
        values = []
        state = self.network.representation(observations)
        for action in actions[:K]:
            state, reward = self.network.dynamics(state, action)
            policy, value = self.network.prediction(state)
            policies.append(policy.squeeze())
            values.append(value.squeeze())
        for k in range(K - len(actions)):
            policies.append(policies[-1])
            values.append(values[-1])
        return policies, values

    def loss(self):
        g = np.random.choice(len(self.REPLAY_BUFFER))
        observations, actions, policies, rewards = self.REPLAY_BUFFER[g]
        move = np.random.choice(len(observations)-2)
        # print(f"starting from move {move} of game {g}")
        with torch.no_grad():
            mcts_policies, _, _ = self.mcts.rollout(observations[move], K=3)
        muzero_policies, muzero_values = self.rollout(observations[move], actions[move:], K=3)
        
        # player1 = zip(mcts_policies[move%2::2], muzero_policies[move%2::2])
        # player2 = zip(mcts_policies[move%2+1::2], muzero_policies[move%2+1::2])

        loss_v = 0
        for v in muzero_values:
            loss_v += (rewards[-1] - v) ** 2
        
        loss_p += torch.dot(torch.tensor(pi), torch.log(p + 1e-9))
        # loss_p = 0
        # for pi, p in player1:
        #     loss_p -= torch.dot(torch.tensor(pi), torch.log(p + 1e-9))
        # for pi, p in player2:
        #     loss_p += torch.dot(torch.tensor(pi), torch.log(p + 1e-9))
        # 
        # if rewards[-1] == 1: # player 1 win
        #     return loss_v + loss_p
        # elif rewards[-1] == -1: # player 2 win
        #     return loss_v - loss_p
        # else: # draw
        #     return loss_v
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
