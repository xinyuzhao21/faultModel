import sys
import matplotlib.pyplot as plt
import numpy as np
# if len(sys.argv)<2:
#     print('No File Name provided')
#     sys.exit()
# fname = sys.argv[1]
fname ='res_7158363.txt'
train_loss = []
valid_loss = []

with open(fname,'r') as f:
    for line in f:
        line = line.split()
        if line[0] == 'accuracy':
            loss = float(line[2])
            train_loss.append(loss)
        if line[0] == 'val_accuracy':
            loss = float(line[2])
            valid_loss.append(loss)
epoches=np.arange(len(train_loss))
plt.plot(epoches,train_loss,'-',label='Train Accu')
plt.plot(epoches,valid_loss,'-',label='Valid Accu')
plt.title('ResNet18: Fine-Tune Full Model')
plt.xlabel( 'Epoches')
plt.ylabel( '%')
plt.legend()

plt.show()