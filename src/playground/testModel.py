import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(1, 1,bias=False)
        with torch.no_grad():
            self.fc1.weight.data[0] = 1
        print('init', self.fc1.weight)
        print()

    def forward(self, x):
        uncorrupted = self.fc1.weight.data.numpy().item()
        result = self.fc1(x)
        print('Forward', x, self.fc1.weight, result)
        print()

        return result

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(3,1,3)
        self.conv.weight.data.fill_(1)
        print('init', self.conv.weight)
        print()

    def forward(self, x):
        out=self.conv(x)
        print('Forward', x, self.conv.weight, out)
        print()

        return out