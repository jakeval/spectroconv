import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class CnnClf(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [5,5,3,3]
        self.channels = [1, 8, 16, 32, 32]

        for i in range(len(self.kernel_sizes)):
          self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1))
          self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
          self.convs.append(nn.ReLU())
          #self.convs.append(nn.Dropout2d(p=0.2))
          self.convs.append(self.pool)
          h = h // 2
          w = w // 2
        print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)
        
        #self.aap = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(self.channels[-1], num_classes)
        #self.fc1 = nn.Linear(h*w*32, 64)
        self.fc1 = nn.Linear(h*w*32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        
    def forward(self, x):
        x = self.convs(x)
        #x = self.aap(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict_from_scores(self, scores):
        return np.argmax(scores, axis=1)

    def predict(self, X, device = None):
        X = torch.tensor(X)
        if device:
          X = X.to(device)
        detached_forward = self.forward(X).detach()
        if device:
          detached_forward = detached_forward.cpu()
        scores = detached_forward.numpy()
        return np.argmax(scores, axis=1)
