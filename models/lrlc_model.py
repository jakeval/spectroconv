import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import lrlc_layer


class LrlcClf(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [parameters.kernel_size] * parameters.num_conv_layers
        self.channels = [1] + [i * parameters.num_channels for i in range(1, parameters.num_conv_layers+1)]

        self.local_dim = parameters.local_dim
        self.rank = parameters.rank
        self.lcw = parameters.lcw # T/F
        self.lrlc_channels = parameters.lrlc_channels # 8

        for i in range(len(self.kernel_sizes)):
            self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1))
            self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
            self.convs.append(nn.ReLU())
            if parameters.dropout > 0:
              self.convs.append(nn.Dropout2d(p=parameters.dropout))
            self.convs.append(self.pool)
            h = h // 2
            w = w // 2
        #print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)
        
        final_num_channels = parameters.num_channels * parameters.num_conv_layers

        self.lrlc_layer = lrlc_layer.LowRankLocallyConnected(self.rank, (h,w), self.channels[-1], self.lrlc_channels, padding='same', local_dim=self.local_dim)
        if self.lcw:
            self.combining_weights = lrlc_layer.LocalizationCombiningWeights(self.rank, (h,w), self.lrlc_layer.L, self.channels[-1], 2, local_dim=self.local_dim)
        else:
            self.combining_weights = lrlc_layer.OuterCombiningWeights(self.rank, self.lrlc_layer.L, local_dim=self.local_dim)

        self.fc1 = nn.Linear(h*w*self.lrlc_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
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
