from torch import nn
import torch


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        super().__init__()
        # print(obs_space_size, action_space_size)

        self.shared_layers = nn.Sequential(
            nn.Linear(210*160, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

        self.value_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # self.shared_layers = nn.Sequential(
        #     nn.Linear(160, 210),
        #     nn.ReLU(),
        #     nn.Linear(210, 160),
        #     nn.ReLU()
        # )
        #
        # self.policy_layers = nn.Sequential(
        #     nn.Linear(160, 210),
        #     nn.ReLU(),
        #     nn.Linear(210, 6)
        # )
        #
        # self.value_layers = nn.Sequential(
        #     nn.Linear(160, 210),
        #     nn.ReLU(),
        #     nn.Linear(210, 1)
        # )

        # self.shared_layers = nn.Sequential(
        #     nn.Linear(obs_space_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU())
        #
        # self.policy_layers = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        #
        # self.value_layers = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1))

    def value(self, obs):
        """Returns value estimate for the state."""
        x = self.shared_layers(obs)
        value = self.value_layers(x)
        return value

    def policy(self, obs):
        """Returns policy logits to sample actions."""
        x = self.shared_layers(obs)
        policy_logits = self.policy_layers(x)
        return policy_logits

    def forward(self, obs):
        """Returns policy logits and value estimate."""
        x = self.shared_layers(obs)
        policy_logits = self.policy_layers(x)
        # print(torch.sum(policy_logits, dim=-1), policy_logits.shape)
        value = self.value_layers(x)
        return policy_logits, value
        # return policy_logits.view(-1, 210*160), value
