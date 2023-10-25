import gymnasium as gym

import torch
from torch import optim
from torch.distributions.categorical import Categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ActorCriticNetwork import ActorCriticNetwork
from PPOTrainer import PPOTrainer
from calc_util import discount_rewards, calculate_gaes
from util import preprocess_img, obs_to_nn_shape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = "ALE/Pong-v5"
# ENVIRONMENT = "PongDeterministic-v4"
# ENVIRONMENT = "ALE/Pong-ram-v5"
EPISODES = 200
LOGGING_FREQ = 5

NOOP = 0
RIGHT = 2
LEFT = 3


def rollout(model, env, max_steps=1000, n_rand=5):
    """
    :param n_rand: number of random actions at the start
    :param model: ActorCriticNetwork
    :param env: Gymnasium Pong-v5
    :param max_steps: max number of steps for rollout
    :return: Training data in the shape (n_steps, obs_shape), reward, observation data
    """
    train_data = [[], [], [], [], []]  # obs, actions, rewards, values, action_log_probs
    obs_data = []
    obs = env.reset()
    # print(env.action_space)
    # print(env.observation_space)
    # gray_frame = cv2.cvtColor(obs[0], cv2.COLOR_BGR2GRAY)
    # print(gray_frame.shape)
    episode_reward = 0

    for i in range(max_steps):
        preprocessed_img = []
        val = 0
        action_log_prob = []
        # if i == 0:
        #     print(obs[0].shape)
        # else:
        #     print(obs.shape)
        # print("len of obs = ", len(obs))

        # take nrand actions at the start
        if i < n_rand:
            action = np.random.choice([RIGHT, LEFT])
            og_action = 0 if RIGHT else 1
        else:
            preprocessed_img = preprocess_img(obs)
            # plt.imshow(preprocessed_img, cmap='Greys')
            # plt.title('preprocessed image')
            # plt.show()

            logits, val = model(torch.tensor([preprocessed_img], dtype=torch.float32, device=DEVICE))
            # logits, val = model(torch.tensor([obs[0]], dtype=torch.float32, device=DEVICE))
            # print(logits, val)
            action_distribution = Categorical(logits=logits)
            # print(action_distribution)
            action = action_distribution.sample()
            # print(action)
            action_log_prob = action_distribution.log_prob(action).item()
            # action_log_prob = action_distribution.probs[0, action].detach().cpu().numpy()
            action, val = action.item(), val.item()

            og_action = action
            # convert action from Action Space of size 3 to RIGHT, LEFT, NOOP
            if action == 0:
                action = RIGHT
            elif action == 1:
                action = LEFT
            else:
                action = NOOP
            # print(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if action != NOOP:
            _, _, terminated, truncated, _ = env.step(NOOP)  # stop moving (NOOP)

        action = og_action

        if terminated or truncated:
            # print("lol")
            break

        obs = next_obs
        episode_reward += reward

        if i >= n_rand:
            for i, item in enumerate((preprocessed_img, action, reward, val, action_log_prob)):
                train_data[i].append(item)
            # gray_frame = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs_data.append(preprocessed_img)

    # print(len(train_data[0]), len(train_data[1]), len(train_data[2]), len(train_data[3]), len(train_data[4]))
    train_data = [np.asarray(x) for x in train_data]
    # max_shape = max([len(arr) for arr in train_data])
    # train_data = [np.reshape(arr, max_shape) for arr in train_data]

    # obtain advantage estimates using rewards and values
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, episode_reward, obs_data


if __name__ == '__main__':
    # env = gym.make(ENVIRONMENT, render_mode="human")
    env = gym.make(ENVIRONMENT)
    # print(env.observation_space.shape, env.action_space.n)
    model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
    model = model.to(DEVICE)
    # train_data, reward = rollout(model, env)
    # print(train_data, reward)
    ppo_trainer = PPOTrainer(model)

    # Hyperparameters
    beta = 0.01

    # Training Loop
    episode_rewards = []
    for episode_idx in range(EPISODES):
        train_data, reward, obs_data = rollout(model, env)
        episode_rewards.append(reward)

        permute_idxs = np.random.permutation(len(train_data[0]))

        # POLICY data
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
        # obs = train_data[0][permute_idxs]

        # GOLDEN DATA
        # obs = torch.tensor([obs_data[0]], dtype=torch.float32, device=DEVICE)
        # print('yolo')
        # poly = model.policy(obs)
        # print('done yolo')

        # permuted_idxs_obs_data = []
        # for p_idx in permute_idxs:
        #     permuted_idxs_obs_data.append(obs_data[p_idx])
        #

        # map obs to new policy logits
        # new_logits = []
        # for i in permute_idxs:
        #     cur_obs = torch.tensor([obs_data[i]], dtype=torch.float32, device=DEVICE)
        #     new_logits.append(model.policy(cur_obs))

        # new_logits = torch.stack(new_logits)

        # map obs to values
        # new_vals = []
        # for i in permute_idxs:
        #     cur_obs = torch.tensor([obs_data[i]], dtype=torch.float32, device=DEVICE)
        #     new_vals.append(model.value(cur_obs))

        # new_vals = torch.stack(new_vals)

        # obs = [torch.from_numpy(obs_array).float().unsqueeze(0).to('cuda:0') for obs_array in obs]
        # obs = torch.from_numpy(obs).float().unsqueeze(0).to('cuda:0')
        # print(len(obs))
        # obs = obs_to_nn_shape(train_data[0])
        # print(obs.shape)
        actions = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
        action_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

        # VALUE data
        returns = torch.tensor(discount_rewards(train_data[2][permute_idxs]), dtype=torch.float32, device=DEVICE)

        # POLICY + VALUE training (train model)
        # ppo_trainer.train_policy(new_logits, actions, action_log_probs, gaes)
        # ppo_trainer.train_value(new_vals, returns)
        ppo_trainer.train_policy(obs.unsqueeze(1), actions, action_log_probs, gaes, beta=beta)
        ppo_trainer.train_value(obs.unsqueeze(1), returns)

        # this reduces exploration in later runs
        beta *= 0.995

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
