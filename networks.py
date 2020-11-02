import torch
import torch.nn as nn
import torch.nn.functional as F


class representation(nn.Module):
    """
    h(observations) = state
      - observations is [b,8,2,3,3] (last 8 board states)
      - state is [b,32,3,3]
    """

    def __init__(self, num_observations: int):
        super().__init__()
        self.conv1 = nn.Conv3d(num_observations, 32, (2, 3, 3), padding=1)  # output becomes [b,32,2,3,3]
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), padding=(0, 1, 1))  # output becomes [b,32,1,3,3]

    def forward(self, observations):
        observations = torch.tensor(observations)
        x = F.relu(self.conv1(observations.unsqueeze(0)))
        state = self.conv2(x).squeeze(2)
        return state


class dynamics(nn.Module):
    """
    g(state, action) = next_state, reward
      - state is [b,32,3,3]
      - action is [b,2,3,3] # includes who's turn it is
      - next_state is [b,32,3,3]
      - reward is [b,1]
    """

    def __init__(self, num_observations:int):
        super().__init__()
        self.conv1 = nn.Conv2d(34, 16, 3, padding=1)  # output is [b, 16, 3, 3]
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)  # output is [b, 16, 3, 3]
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # output is [b, 32, 3, 3] (next state)
        self.dense1 = nn.Linear(16 * 3 * 3, num_observations * 3 * 3)  # output is [b, 8*3*3]
        self.dense2 = nn.Linear(num_observations * 3 * 3, 1)  # output is [b, 1] (reward)

    def forward(self, state, action):
        x = F.relu(self.conv1(torch.cat([state, action], 1)))
        x = F.relu(self.conv2(x))
        next_state = self.conv3(x)

        x = F.relu(self.dense1(torch.flatten(x, start_dim=1)))
        reward = torch.tanh(self.dense2(x))

        return next_state, reward


class prediction(nn.Module):
    """
    f(state) = policy, value
      - state is [b,32,3,3]
      - policy is [b,9]
      - value is [b,1]
    """

    @property
    def policy_dim(self):
        return 9

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 16, 3, padding=1)  # output is [b,16,3,3]
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)  # output is [b,8,3,3]
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)  # output is [b,8,3,3]
        self.dense1 = nn.Linear(8 * 3 * 3, 3 * 3)
        self.dense2 = nn.Linear(8 * 3 * 3, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8 * 3 * 3)
        policy = F.softmax(self.dense1(x), dim=1)
        value = torch.tanh(self.dense2(x))

        return policy, value
