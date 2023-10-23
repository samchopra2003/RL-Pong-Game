import numpy as np
import torch


def preprocess_img(image, bkg_color=np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img

def obs_to_nn_shape(obs):
    """
    :param obs: Tensor of observations
    :return: Compatible tensor with NN
    """
    obs = [torch.from_numpy(obs_array) for obs_array in obs]
    obs = torch.stack(obs)
    policy_input = obs.view(-1, *obs.shape[-1:])
    # print(policy_input.shape)
    return policy_input

