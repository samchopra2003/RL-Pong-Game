import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_preproc = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                     axis=-1) / 255.
    batch_input = np.swapaxes(list_of_images_preproc, 0, 1)
    return torch.from_numpy(batch_input).float().to(DEVICE)
