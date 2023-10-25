from torch import nn
import torch
from torch import optim
from torch.distributions.categorical import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPO_CLIP_VAL = 0.2
TARGET_KL_DIV = 0.01
MAX_POLICY_TRAIN_ITERS = 5
VALUE_TRAIN_ITERS = 5
POLICY_LR = 3e-4
VALUE_LR = 1e-3


class PPOTrainer:
    def __init__(self,
                 actor_critic,
                 ppo_clip_val=PPO_CLIP_VAL,
                 target_kl_div=TARGET_KL_DIV,
                 max_policy_train_iters=MAX_POLICY_TRAIN_ITERS,
                 value_train_iters=VALUE_TRAIN_ITERS,
                 policy_lr=POLICY_LR,
                 value_lr=VALUE_LR
                 ):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        policy_params = list(self.ac.shared_layers.parameters()) + \
                        list(self.ac.policy_layers.parameters())
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

        value_params = list(self.ac.shared_layers.parameters()) + \
                       list(self.ac.policy_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, actions, old_log_probs, gaes, beta=0.01):
        """
        :param new_logits: new_logits mapped to observations
        :param actions: actions
        :param old_log_probs: old log probabilities
        :param gaes: General Advantgae Estimations
        :param beta: Coefficient for entropy (exploration)
        :return:
        """
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()  # reset gradient

            # new_logits = []
            # for i in range(len(obs)):
            #     cur_obs = torch.tensor([obs[i]], dtype=torch.float32, device=DEVICE)
            #     new_logits.append(self.ac.policy(obs))

            # new_logits = self.ac.policy(torch.tensor([obs], dtype=torch.float32, device=DEVICE))

            # new_logits = self.ac.policy(obs[0].unsqueeze(0))
            new_logits = self.ac.policy(obs)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(actions)

            # print(new_logits)

            # new_log_probs = []
            # for i in range(len(new_logits)):
            #     cur_logits = Categorical(logits=new_logits[i])
            #     new_log_probs.append(cur_logits.log_prob(actions[i]))

            # new_log_probs = torch.stack(new_log_probs)

            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes

            old_probs = torch.exp(old_log_probs)
            new_probs = torch.exp(new_log_probs)
            # include a regularization term
            # this steers new_policy towards 0.5
            # add in 1.e-10 to avoid log(0) which gives nan
            entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                        (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))
            # negative to turn this to loss
            # policy_loss = (-torch.min(full_loss, clipped_loss)).mean()
            policy_loss = (-(torch.min(full_loss, clipped_loss) + beta * entropy)).mean()

            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.ac.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()
