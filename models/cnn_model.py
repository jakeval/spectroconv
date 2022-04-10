import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class CnnClf(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        h, w = input_shape
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 8, 5, padding='same', stride=1)
        h = h // 2
        w = w // 2
        self.conv2 = nn.Conv2d(8, 16, 5, padding='same', stride=1)
        h = h // 2
        w = w // 2
        self.conv3 = nn.Conv2d(16, 16, 3, padding='same', stride=1)
        h = h // 2
        w = w // 2
        self.fc1 = nn.Linear(h*w*16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, X):
        X = torch.tensor(X)
        scores = self.forward(X).detach().numpy()
        return np.argmax(scores, axis=1)
