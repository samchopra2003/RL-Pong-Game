from torch import nn
import torch


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        """
        :param obs_space_size: 1 channel
        :param action_space_size: pre_processed img size (grayscale)
        """
        super().__init__()

        # 80x80x1 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x16
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16

        # two fully connected layers
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 3)

        self.shared_layers = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU()
        )

        self.policy_layers = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

        self.value_layers = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def value(self, obs):
        """Returns value estimate for the state."""
        x = self.shared_layers(obs)
        x = x.view(-1, self.size)
        value = self.value_layers(x)
        return value

    def policy(self, obs):
        """Returns policy logits to sample actions."""
        x = self.shared_layers(obs)
        x = x.view(-1, self.size)
        policy_logits = self.policy_layers(x)
        return policy_logits

    def forward(self, obs):
        """Returns policy logits and value estimate."""
        x = self.shared_layers(obs)
        x = x.view(-1, self.size)
        policy_logits = self.policy_layers(x)
        value = self.value_layers(x)
        return policy_logits, value
