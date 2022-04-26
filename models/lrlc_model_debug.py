import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import lrlc_layer


class LrlcClfDebug(nn.Module):
    def __init__(self, rank, input_shape, class_enums, local_dim=None, lcw=False):
        super().__init__()
        h, w = input_shape
        
        self.rank = rank
        self.convs = []
        self._conv_layers = []
        self._conv_layer_outputs = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [5,5,3]
        self.channels = [1, 8, 16, 32]
        self.local_dim = local_dim
        self.lcw = lcw

        for i in range(len(self.kernel_sizes)):
            self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1))
            self._conv_layers.append(self.convs[-1])
            self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
            self.convs.append(nn.ReLU())
            #self.convs.append(nn.Dropout2d(p=0.2))
            self.convs.append(self.pool)
            self._conv_layer_outputs.append([h, w])
            h = h // 2
            w = w // 2
        #print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)

        lrlc_channels = 8
        self.lrlc_layer = lrlc_layer.LowRankLocallyConnected(self.rank, (h,w), self.channels[-1], lrlc_channels, padding='same', local_dim=local_dim)
        if lcw:
            self.combining_weights = lrlc_layer.LocalizationCombiningWeights(self.rank, (h,w), self.lrlc_layer.L, self.channels[-1], 2, local_dim=local_dim)
        else:
            self.combining_weights = lrlc_layer.OuterCombiningWeights(self.rank, self.lrlc_layer.L, local_dim=local_dim)

        #self.aap = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(self.channels[-1], num_classes)
        #self.fc1 = nn.Linear(h*w*32, 64)
        self.fc1 = nn.Linear(h*w*lrlc_channels, 64)
        self.fc2 = nn.Linear(64, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
        cw = None
        if self.lcw:
            cw = self.combining_weights(x)
        else:
            cw = self.combining_weights()
        x = F.relu(self.lrlc_layer(x, cw))
        x = torch.flatten(x, 1)
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

    def loss(self, S, y, **kwargs):
        return F.cross_entropy(S, y, **kwargs)

    def ordinal_to_class_enum(self, y):
        y2 = [None] * len(y)
        y = y.squeeze()
        for i in range(len(y)):
            y2[i] = self.class_names[y[i]]
        return y2

    def get_weights(self):
        return self.lrlc_layer.get_weight(self.combining_weights())
