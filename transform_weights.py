import torch
import torch.nn as nn
import numpy as np

net = torch.load('weights.pth')
for key, v in net.items():
    if 'conv' in key:
        v = v.view(v.shape[0]*v.shape[1], v.shape[2] * v.shape[3])
        print('here')
    weight_file = str(key)+'.txt'
    weights = v.numpy()
    with open(weight_file, 'w') as file:
        np.savetxt(file, weights)
