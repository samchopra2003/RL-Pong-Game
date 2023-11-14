from torch import nn
import torch.nn.functional as F

import cv2


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=6, stride=2)
        self.size = 8 * 15 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("Input to conv1: ", x.shape)
        x = F.relu(self.conv1(x))
        # print("Input to conv2: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("Output of conv2: ", x.shape)
        x = F.relu(self.conv3(x))
        # print("Output of conv3: ", x.shape)
        x = x.view(-1, self.size)
        # print("Input to fc1: ", x.shape)
        x = F.relu(self.fc1(x))
        # print("Input to fc2: ", x.shape)
        # return self.sig(self.fc2(x))
        # x = self.fc2(x)
        # print("Output of network: ", x.shape)
        return self.sig(self.fc2(x))
        # x = F.softmax(x, dim=1)
        # return x
