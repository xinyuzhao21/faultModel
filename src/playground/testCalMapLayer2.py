import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1,bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.fc1.weight.data[0] = 1
            self.fc2.weight.data[0] = 2
        print('init', self.fc1.weight,self.fc2.weight)
        print()

    def forward(self, x):
        uncorrupted = self.fc1.weight.data.numpy().item()
        print('uncorrupted',uncorrupted)

        z1 = self.fc1(x)
        a1 = self.relu(z1)
        with torch.no_grad():
            self.fc1.weight.data[0] = uncorrupted
        z2 = self.fc2(a1)
        print('Forward', x, z1,a1,z2)
        print("weights",self.fc1.weight,self.fc2.weight)
        print()

        return z2



model = Net()
criterion = lambda x, y: x - y
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0

    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = torch.tensor([2], dtype=torch.float), torch.tensor([0])

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    print('grad', model.fc1.weight.grad,model.fc2.weight.grad)
    print("loss",loss)
    optimizer.step()

    # print statistics
    running_loss += loss.item()

print('Finished Training')