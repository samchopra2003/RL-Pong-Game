from torch import nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #
    #     self.conv1 = nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False)
    #     self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
    #     self.size = 16 * 39 * 63
    #
    #     # two fully connected layers
    #     self.fc1 = nn.Linear(self.size, 256)
    #     self.fc2 = nn.Linear(256, 3)
    #
    #     self.shared_layers = nn.Sequential(
    #         self.conv1,
    #         nn.ReLU(),
    #         self.conv2,
    #         nn.ReLU()
    #     )
    #
    #     self.policy_layers = nn.Sequential(
    #         self.fc1,
    #         nn.ReLU(),
    #         self.fc2,
    #     )
    #
    #     self.value_layers = nn.Sequential(
    #         self.fc1,
    #         nn.ReLU(),
    #         nn.Linear(256, 1)
    #     )
    #
    # def value(self, obs):
    #     """Returns value estimate for the state."""
    #     x = self.shared_layers(obs)
    #     x = x.view(-1, self.size)
    #     value = self.value_layers(x)
    #     return value
    #
    # def policy(self, obs):
    #     """Returns policy logits to sample actions."""
    #     x = self.shared_layers(obs)
    #     x = x.view(-1, self.size)
    #     policy_logits = self.policy_layers(x)
    #     return policy_logits
    #
    # def forward(self, obs):
    #     """Returns policy logits and value estimate."""
    #     x = self.shared_layers(obs)
    #     x = x.view(-1, self.size)
    #     policy_logits = self.policy_layers(x)
    #     value = self.value_layers(x)
    #     return policy_logits, value

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 16 * 11 * 21

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("Input to conv1: ", x.shape)
        x = F.relu(self.conv1(x))
        # print("Input to conv2: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("Output of conv2: ", x.shape)
        x = x.view(-1, self.size)
        # print("Input to fc1: ", x.shape)
        x = F.relu(self.fc1(x))
        # print("Input to fc2: ", x.shape)
        # return self.sig(self.fc2(x))

        x = self.sig(self.fc2(x))
        # print("Output of network: ", x.shape)
        return x
