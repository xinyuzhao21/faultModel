import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
class LinearNet(nn.Module):
    def __init__(self,inclass=1,allone=1):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(inclass, 1,bias=False)
        self.fc1.weight.data.fill_(allone)
        # print('init', self.fc1.weight)
        # print()

    def forward(self, x):
        result = self.fc1(x)
        # print('Forward', x, self.fc1.weight, result)
        # print()

        return result

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(3,1,3)
        # self.conv.weight.data.fill_(1)
        print('init', self.conv.weight)
        print()

    def forward(self, x):
        out=self.conv(x)
        print(out.shape,out)
        print('Forward', self.conv.weight, out)
        print()

        return out