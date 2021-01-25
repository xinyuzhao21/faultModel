import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
# if len(sys.argv)<2:
#     print('No File Name provided')
#     sys.exit()
# fname = sys.argv[1]
fname = 'res_7158363.txt'

def parse_dir(dirname,ax1,ax2):
    for filename in glob.glob(os.path.join(dirname, '*.txt')):
        print(filename)
        train_loss, valid_loss,train_accuracy,valid_accuracy,epoches = parse_file(filename)
        plot(ax1,epoches,valid_loss,{'label':filename})
        plot(ax2, epoches, valid_accuracy, {'label': filename})

def parse_file(fname):
    train_loss = []
    train_accuracy = []
    valid_accuracy = []
    valid_loss = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.split()
            if line[0] == 'accuracy':
                loss = float(line[2])
                train_accuracy.append(loss)
            if line[0] == 'loss':
                loss = float(line[2])
                train_loss.append(loss)
            if line[0] == "val_loss":
                loss = float(line[2])
                valid_loss.append(loss)
            if line[0] == 'val_accuracy':
                loss = float(line[2])
                valid_accuracy.append(loss)
    epoches = np.arange(len(train_loss))
    return train_loss, valid_loss,train_accuracy,valid_accuracy,epoches

def plot(ax,x,y,param_dict):
    out = ax.plot(x, y, **param_dict)
    return out

fig,(ax1,ax2) = plt.subplots(1,2)
parse_dir('out/selected/1-25',ax1,ax2)
plt.title('ResNet18')
plt.xlabel('Epoches')
plt.ylabel('%')
# plt.legend()
plt.show()
