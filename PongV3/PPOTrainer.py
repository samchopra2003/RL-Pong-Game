import torch
from torch import optim
import numpy as np

from helper_utils.states_to_prob import states_to_prob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPO_CLIP_VAL = 0.1
TARGET_KL_DIV = 0.01
MAX_POLICY_TRAIN_ITERS = 3
POLICY_LR = 2.5e-4
BETA = 0.01

RIGHTFIRE = 0
LEFTFIRE = 1


class PPOTrainer:
    def __init__(self,
                 policy_net,
                 target_kl_div=TARGET_KL_DIV,
                 max_policy_train_iters=MAX_POLICY_TRAIN_ITERS,
                 policy_lr=POLICY_LR
                 ):
        self.policy_net = policy_net
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters

        self.policy_optim = optim.Adam(policy_net.parameters(), lr=policy_lr)

    def train_policy(self, old_probs, states, actions, rewards, discount=0.99, epsilon=0.1, beta=0.01):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()  # reset gradient

            cur_discount = discount ** np.arange(len(rewards))
            cur_rewards = np.array(rewards)
            cur_rewards = cur_rewards * cur_discount[:, np.newaxis]

            # convert rewards to future rewards
            rewards_future = cur_rewards[::-1].cumsum(axis=0)[::-1]

            mean = np.mean(rewards_future, axis=1)
            std = np.std(rewards_future, axis=1) + 1.0e-10

            rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

            # convert everything into pytorch tensors and move to gpu if available
            cur_actions = torch.tensor(actions, dtype=torch.int8, device=DEVICE)
            cur_old_probs = torch.tensor(old_probs, dtype=torch.float32, device=DEVICE)
            cur_rewards = torch.tensor(rewards_normalized, dtype=torch.float32, device=DEVICE)

            # convert states to policy (or probability)
            new_probs = states_to_prob(self.policy_net, states)
            new_probs = torch.where(cur_actions == RIGHTFIRE, new_probs, 1.0 - new_probs)

            # ratio for clipping
            ratio = new_probs / cur_old_probs

            # clipped function
            clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            clipped_surrogate = torch.min(ratio * cur_rewards, clip * cur_rewards)

            # include a regularization term
            # this steers new_policy towards 0.5
            # add in 1.e-10 to avoid log(0) which gives nan
            entropy = -(new_probs * torch.log(cur_old_probs + 1.e-10) +
                        (1.0 - new_probs) * torch.log(1.0 - cur_old_probs + 1.e-10))

            # this returns an average of all the entries of the tensor
            # effective computing L_sur^clip / T
            # averaged over time-step and number of trajectories
            # this is desirable because we have normalized our rewards
            policy_loss = -torch.mean(clipped_surrogate + beta * entropy)

            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (torch.log(cur_old_probs + 1.e-10) - torch.log(new_probs + 1.e-10)).mean()
            if kl_div >= self.target_kl_div:
                break
