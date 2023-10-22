import gymnasium as gym

import torch
from torch import optim
from torch.distributions.categorical import Categorical
import numpy as np
import cv2

from ActorCriticNetwork import ActorCriticNetwork
from PPOTrainer import PPOTrainer
from util import discount_rewards, calculate_gaes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = "ALE/Pong-v5"
# ENVIRONMENT = "ALE/Pong-ram-v5"
EPISODES = 10
LOGGING_FREQ = 10


def rollout(model, env, max_steps=1000):
    """
    :param model: ActorCriticNetwork
    :param env: Gymnasium Pong-v5
    :param max_steps: max number of steps for rollout
    :return: Training data in the shape (n_steps, obs_shape)
    """
    train_data = [[], [], [], [], []]  # obs, actions, rewards, values, action_log_probs
    obs = env.reset()
    # print(env.action_space)
    # print(env.observation_space)
    gray_frame = cv2.cvtColor(obs[0], cv2.COLOR_BGR2GRAY)
    # print(gray_frame.shape)
    episode_reward = 0
    for _ in range(max_steps):
        # gray_frame = cv2.cvtColor(obs[0], cv2.COLOR_BGR2GRAY)
        gray_frame_flattened = gray_frame.reshape(-1)
        logits, val = model(torch.tensor([gray_frame_flattened], dtype=torch.float32, device=DEVICE))
        # logits, val = model(torch.tensor([obs[0]], dtype=torch.float32, device=DEVICE))
        # print(logits, val)
        action_distribution = Categorical(logits=logits)
        # print(action_distribution)
        action = action_distribution.sample()
        # print(action)
        action_log_prob = action_distribution.log_prob(action).item()
        # action_log_prob = action_distribution.probs[0, action].detach().cpu().numpy()
        action, val = action.item(), val.item()
        # print(action)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        for i, item in enumerate((gray_frame_flattened, action, reward, val, action_log_prob)):
            train_data[i].append(item)

        obs = next_obs
        gray_frame = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        episode_reward += reward

    # print(len(train_data[0]), len(train_data[1]), len(train_data[2]), len(train_data[3]), len(train_data[4]))
    train_data = [np.asarray(x) for x in train_data]
    # max_shape = max([len(arr) for arr in train_data])
    # train_data = [np.reshape(arr, max_shape) for arr in train_data]

    # obtain advantage estimates using rewards and values
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, episode_reward


if __name__ == '__main__':
    # env = gym.make(ENVIRONMENT, render_mode="human")
    env = gym.make(ENVIRONMENT)
    # print(env.observation_space.shape, env.action_space.n)
    model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
    model = model.to(DEVICE)
    # train_data, reward = rollout(model, env)
    # print(train_data, reward)
    ppo_trainer = PPOTrainer(model)

    # Training Loop
    episode_rewards = []
    for episode_idx in range(EPISODES):
        train_data, reward = rollout(model, env)
        episode_rewards.append(reward)

        permute_idxs = np.random.permutation(len(train_data[0]))

        # POLICY data
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
        action_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

        # VALUE data
        returns = torch.tensor(discount_rewards(train_data[2][permute_idxs]), dtype=torch.float32, device=DEVICE)

        # POLICY + VALUE training (train model)
        ppo_trainer.train_policy(obs, actions, action_log_probs, gaes)
        ppo_trainer.train_value(obs, returns)

        if (episode_idx + 1) % LOGGING_FREQ == 0:
            print('Episode {} | Avg. Rewards {:.1f}'.format(
                episode_idx + 1, np.mean(episode_rewards[-LOGGING_FREQ:])
            ))

    # save policy
    torch.save(model, 'PPO_pong.pkl')




#     observation, info = env.reset()
#     for _ in range(1000):
#         action = env.action_space.sample()  # agent policy that uses the observation and info
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         if terminated or truncated:
#             observation, info = env.reset()
#
#     env.close()
