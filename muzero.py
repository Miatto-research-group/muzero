import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from games import TicTacToe
from mcts import Tree

np.set_printoptions(suppress=False, linewidth=280)


class representation(nn.Module):
    """
    h(observations) = state
      - observations is [8,3,3] (last 8 board states)
      - state is [32,3,3]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1) # padding by 1 so output planes remain 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, observations):
        observations = torch.stack([torch.zeros([8 - observations.shape[0], 3, 3]), observations], 0)
        x = F.relu(self.conv1(observations.unsqueeze(0)))
        state = F.relu(self.conv2(x))
        return state

class dynamics(nn.Module):
    """
    g(state, action) = next_state, reward
      - state is [32,3,3]
      - action is [3,3]
      - next_state is [32,3,3]
      - reward is [1]

    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(33, 16, 3, padding=1) # padding by 1 so output planes remain 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dense1 = nn.Linear(8*3*3, 3*3)
        self.dense2 = nn.Linear(3*3, 1) # TODO: not this?

    def forward(self, state, action):
        x = F.relu(self.conv1(torch.stack([state, action], 1)))
        next_state = F.relu(self.conv2(x.unsqueeze(0)))

        x = F.relu(self.dense1(torch.flatten(x, start_dim=1))) # keep batch
        reward = F.tanh(self.dense2(x))

        return next_state, reward

class prediction(nn.Module):
    """
    f(state) = policy, value
      - state is [32,3,3]
      - policy is [3,3]
      - value is [1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 8, 3, padding=1) # padding by 1 so output planes remain 3x3
        self.conv2 = nn.Conv2d(8, 1, 3, padding=1)
        self.dense1 = nn.Linear(8*3*3, 3*3)
        self.dense2 = nn.Linear(3*3, 1) # TODO: not this?

    def forward(self, state):
        x = F.relu(self.conv1(state))
        policy = F.relu(self.conv2(x))

        x = F.relu(self.dense1(x.view(-1, 8*3*3)))
        value = F.tanh(self.dense2(x))

        return policy, value


    @staticmethod
    def _sample(mask, policy):
        policy = policy * mask
        if np.allclose(policy, 0):  # if no legal moves are available
            policy = mask
        return np.random.choice(len(policy), p=policy / np.sum(policy))


class Muzero:


    def __init__(self, Game):
        self.REPLAY_BUFFER = []
        self.game = Game()
        self.MCTS = MCTS()

    # A
    def MCTS_search(self, observations: list, num_simulations: int):
        self.MCTS.reset()
        self.MCTS.initialize(self.representation(observations), self.model)
        policy, value = self.MCTS.search(num_simulations)
        return policy, value

    # B
    def MCTS_play(self):
        self.game.__init__()
        observations = self.game.observations
        MCTS_policies = []
        actions = []
        illegal = 0
        while not self.game.end:
            MCTS_policy, MCTS_value = self.MCTS_search(game.observations, num_simulations=50)
            MCTS_policy = MCTS_policy * self.game.legal_moves_mask
            if np.allclose(MCTS_policy, 0):
                illegal += 1
                continue
            action_index = np.random.choice(self.game.num_actions, p = MCTS_policy.reshape(-1) / np.sum(MCTS_policy))
            observed_reward = self.game.play(self.game.action(action_index))  # at the end of the game, observed_reward contains the final value (+1 win, -1 loss, 0 graw)
            actions.append(action)
            MCTS_policies.append(MCTS_policy)
            observations.append(self.game.observations)
        if illegal > 0:
            print(f"skipped {illegal} illegal moves")
        return observations, actions, MCTS_policies, observed_reward

    # Model rollout (C)
    def model_rollout(self, observations: list, actions: list):
        state = self.representation(observations)
        
        model_policies = []
        model_values = []
        for action in actions:  # TODO: for a in actions
            state, policy, value, reward = self.model(state, action)
            model_policies.append(tf.squeeze(policy))
            model_values.append(tf.squeeze(value))
        return model_policies, model_values

    def loss(self):
        observations, actions, MCTS_policies, observed_reward = self.MCTS_play()
        self.REPLAY_BUFFER.append((observations, actions, MCTS_policies, observed_reward))
        model_policies, model_values = self.model_rollout(observations, actions)
        loss_v = 0
        loss_p = 0
        for i in range(min(5, len(model_policies))):  # TODO: for i in len(policies_p)
            loss_p += k.losses.categorical_crossentropy(MCTS_policies[i], model_policies[i]))
            loss_v += (observed_reward - model_values[i]) ** 2
        return loss_p + loss_v

    @property
    def trainable_variables(self):
        return (
            self.policy.trainable_variables
            + self.value.trainable_variables
            + self.representation.trainable_variables
            + self.new_state.trainable_variables
        )  # not reward for games
