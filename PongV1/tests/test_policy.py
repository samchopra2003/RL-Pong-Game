import torch
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np

from PongV1.helper_utils.preprocess_imgs import preprocess_img

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = "ALE/Pong-v5"

NOOP = 0
RIGHT = 2
LEFT = 3

# Load the pre-trained policy parameters
# policy = torch.load('../PPO_pong.pkl')
policy = torch.load('../PPO.policy')

def test_model(model, env, max_steps=1000, n_rand=5):
    """
    :param n_rand: number of random actions at the start
    :param model: ActorCriticNetwork
    :param env: Gymnasium Pong-v5
    :param max_steps: max number of steps for rollout
    :return: Training data in the shape (n_steps, obs_shape), reward, observation data
    """
    obs = env.reset()
    episode_reward = 0

    for i in range(max_steps):
        if i < n_rand:
            action = np.random.choice([RIGHT, LEFT])
        else:
            preprocessed_img = preprocess_img(obs)

            logits, val = model(torch.tensor([preprocessed_img], dtype=torch.float32, device=DEVICE))
            action_distribution = Categorical(logits=logits)
            action = action_distribution.sample()
            action_log_prob = action_distribution.log_prob(action).item()
            action, val = action.item(), val.item()

            # convert action from Action Space of size 3 to RIGHT, LEFT, NOOP
            if action == 0:
                action = RIGHT
            elif action == 1:
                action = LEFT
            else:
                action = NOOP

        next_obs, reward, terminated, truncated, info = env.step(action)
        # if action != NOOP:
        _, _, terminated, truncated, _ = env.step(NOOP)  # stop moving (NOOP)

        if terminated or truncated:
            # print("lol")
            break

        obs = next_obs


if __name__ == '__main__':
    env = gym.make(ENVIRONMENT, render_mode="human")
    # env = gym.make(ENVIRONMENT)
    # model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
    # model = model.to(DEVICE)
    with torch.no_grad():
        test_model(policy, env)
