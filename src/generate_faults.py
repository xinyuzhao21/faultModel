from playground import corrupt
import torch
conv1 =  torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias =False)
print(conv1.weight.shape)
total = 1
for d in conv1.weight.shape:
    total*=d
p = 0.01
faults = corrupt.Fault.generate_faults(*list(conv1.weight.shape),sample_size=int(total*p))
