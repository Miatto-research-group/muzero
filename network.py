import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


class Representation(nn.Module):
    """
    h(observations) = state
      - observations is [N,2,3,3] (last N board states)
      - state is [32,3,3]
    """
    def __init__(self, num_observations: int):
        super().__init__()
        self.conv1 = nn.Conv3d(num_observations, 32, (2, 3, 3), padding=1)  # output becomes [1,32,2,3,3]
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), padding=(0, 1, 1))  # output becomes [1,32,1,3,3]

    def forward(self, observations):
        observations = torch.tensor(observations, dtype=torch.float32)
        x = F.relu(self.conv1(observations.unsqueeze(0)))
        state = self.conv2(x).squeeze()
        return state


class Dynamics(nn.Module):
    """
    g(state, action) = next_state, reward
      - state is [32,3,3]
      - action is [3,3]
      - next_state is [32,3,3]
      - reward is [1]
    """
    def __init__(self, num_observations:int):
        super().__init__()
        self.conv1 = nn.Conv2d(33, 16, 3, padding=1)  # output is [1, 16, 3, 3]
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)  # output is [1, 16, 3, 3]
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # output is [1, 32, 3, 3] (next state)
        self.dense1 = nn.Linear(16 * 3 * 3, num_observations * 3 * 3)  # output is [1, 8*3*3]
        self.dense2 = nn.Linear(num_observations * 3 * 3, 1)  # output is [1, 1] (reward)

    def forward(self, state, action):
        x = F.relu(self.conv1(torch.cat([state.unsqueeze(0), torch.tensor(action[None,...], dtype=torch.float32).unsqueeze(0)], 1)))
        x = F.relu(self.conv2(x))
        next_state = self.conv3(x).squeeze(0)

        x = F.relu(self.dense1(torch.flatten(x, start_dim=1)))
        reward = torch.tanh(self.dense2(x)).squeeze(0)

        return next_state, reward

class Prediction(nn.Module):
    """
    f(state) = policy, value
      - state is [32,3,3]
      - policy is [9]
      - value is [1]
    """
    @property
    def policy_dim(self):
        return 9

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 16, 3, padding=1)  # output is [1,16,3,3]
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)  # output is [1,8,3,3]
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)  # output is [1,8,3,3]
        self.dense1 = nn.Linear(8 * 3 * 3, 3 * 3)
        self.dense2 = nn.Linear(8 * 3 * 3, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state.unsqueeze(0)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8 * 3 * 3)
        policy = F.softmax(self.dense1(x), dim=1).squeeze(0)
        value = torch.tanh(self.dense2(x)).squeeze(0)

        return policy, value


class Network:
    def __init__(self, num_observations: int):
        self.prediction = Prediction()
        self.dynamics = Dynamics(num_observations)
        self.representation = Representation(num_observations)

    @property
    def parameters(self):
        return chain(self.representation.parameters(), self.dynamics.parameters(), self.prediction.parameters())