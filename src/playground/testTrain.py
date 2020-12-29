import torch.optim as optim
import torch
import playground.corrupt as corrupt
def train(model):
    criterion = lambda x, y: x - y
    optimizer = optim.SGD(model.parameters(), lr=1)

    for epoch in range(2):  # loop over the dataset multiple times
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = torch.tensor([2], dtype=torch.float), torch.tensor([0])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        print('grad', model.fc1.weight.grad)
        print("loss", loss)
        optimizer.step()

def train_corrupt(model,fault):
    criterion = lambda x, y: x - y
    optimizer = optim.SGD(model.parameters(), lr=1)

    for epoch in range(2):  # loop over the dataset multiple times
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = torch.ones((1,3,3,3), dtype=torch.float), torch.tensor([0])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if not fault.injected:
            origin,corrupted=corrupt.FaultInject.weight_inject(fault,model)
            print('Origin',origin,'Corrupted',corrupted)
            outputs = model(inputs)
            fault.corrupt_value=origin
            origin,corrupted=corrupt.FaultInject.weight_inject(fault,model)
            print('Origin', origin, 'Corrupted', corrupted)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        print('grad',epoch)
        print("loss",epoch, loss)
        optimizer.step()