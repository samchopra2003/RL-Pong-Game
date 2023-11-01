import numpy as np
import torch
import progressbar as pb

from parallelEnv import parallelEnv
from collect_rollouts import collect_rollouts
from PolicyNetwork import PolicyNetwork
from PPOTrainer import PPOTrainer

# GLOBAL VARS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 2000
LOGGING_FREQ = 10
ENV_NAME = 'PongDeterministic-v4'

widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=EPISODES).start()

# hyperparameters
discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320

reward_file = './rewards.txt'

if __name__ == "__main__":
    ENVS = parallelEnv(ENV_NAME, n=8, seed=1234)
    # keep track of progress
    mean_rewards = []

    policy = PolicyNetwork().to(DEVICE)
    ppoTrainer = PPOTrainer(policy)

    # clear previous contents
    with open(reward_file, 'w'):
        pass

    # training loop
    for episode_idx in range(EPISODES):

        # collect trajectories
        old_probs, states, actions, rewards = \
            collect_rollouts(ENVS, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # train POLICY
        ppoTrainer.train_policy(old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress
        if (episode_idx + 1) % LOGGING_FREQ == 0:
            print("Episode: {0:d}, score: {1:f}".format(episode_idx + 1, np.mean(total_rewards)))
            print(total_rewards)

        # append mean reward to text file
        with open(reward_file, 'a') as file:
            file.write(str(np.mean(total_rewards)) + '\n')

        # update progress widget bar
        timer.update(episode_idx + 1)
        # save policy
        if (episode_idx + 1) % 500 == 0:
            torch.save(policy, 'PPO_deterministic.policy')

    torch.save(policy, 'PPO_deterministic_final.policy')

    timer.finish()