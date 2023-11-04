import numpy as np
import torch

from helper_utils.preprocess_img import preprocess_batch

# the 'FIRE' part ensures that the game starts again after losing a life
NOOP = 0
RIGHTFIRE = 4
LEFTFIRE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_rollouts(envs, policy, tmax=200, nrand=5):
    # number of parallel instances
    n = len(envs.ps)

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    envs.reset()

    # start all parallel agents
    envs.step([1] * n)

    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHTFIRE, LEFTFIRE], n))
        fr2, re2, _, _ = envs.step([0] * n)

    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1, fr2])

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = np.where(np.random.rand(n) < probs, RIGHTFIRE, LEFTFIRE)
        probs = np.where(action == RIGHTFIRE, probs, 1.0 - probs)

        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([NOOP] * n)

        reward = re1 + re2

        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list
