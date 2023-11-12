import torch


def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])
