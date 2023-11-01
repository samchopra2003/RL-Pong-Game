import torch
import gymnasium as gym
from PongV2.helper_utils.preprocess_img import preprocess_batch
import numpy as np
import random as rand

NOOP = 0
RIGHTFIRE = 4
LEFTFIRE = 5

def play(env, policy, time=2000, preprocess=None, nrand=5):
    env.reset()

    # star game
    env.step(1)

    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, _, terminated, truncated, _ = env.step(np.random.choice([RIGHTFIRE, LEFTFIRE]))
        frame2, _, terminated, truncated, _ = env.step(0)

    anim_frames = []

    for _ in range(time):

        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)

        # RIGHT = 4, LEFT = 5
        action = RIGHTFIRE if rand.random() < prob else LEFTFIRE
        frame1, _, terminated, truncated, _ = env.step(action)
        frame2, _, terminated, truncated, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if terminated or truncated:
            break

    env.close()

    return


# policy = torch.load('PPO_deterministic.policy')
# env = gym.make('PongDeterministic-v4', render_mode="human")

policy = torch.load('PPO_stochastic.policy')
env = gym.make("ALE/Pong-v5", render_mode="human")

with torch.no_grad():
    play(env, policy, time=2000)