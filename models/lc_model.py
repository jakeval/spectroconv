import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import local_layer


class LcClf(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.pool = nn.MaxPool2d(2,2)

        if parameters.num_conv_layers == 1:
          self.kernel_sizes = [parameters.kernel_size_start]
          self.channels = [1, parameters.final_filter_size]

        if parameters.num_conv_layers == 2:
          self.kernel_sizes = [parameters.kernel_size_start, parameters.kernel_size_start // 2]
          self.channels = [1, 8, parameters.final_filter_size]

        elif parameters.num_conv_layers == 3:
          self.kernel_sizes = [parameters.kernel_size_start, parameters.kernel_size_start, parameters.kernel_size_start // 2]
          self.channels = [1, 8, parameters.final_filter_size//2, parameters.final_filter_size]

        elif parameters.num_conv_layers == 4:
          self.kernel_sizes = [parameters.kernel_size_start, parameters.kernel_size_start, parameters.kernel_size_start//2, parameters.kernel_size_start//2]
          self.channels = [1, 8, parameters.final_filter_size//2, parameters.final_filter_size, parameters.final_filter_size]

        self.local_dim = parameters.local_dim

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

        if self.local_dim is not None:
            self.lc_layer = local_layer.LocallyConnectedConv(self.local_dim, (h,w) , parameters.final_filter_size, parameters.final_filter_size, padding='same')
        else:
            self.lc_layer = local_layer.LocallyConnected((h,w) , parameters.final_filter_size, parameters.final_filter_size, padding='same')
        
        self.fc1 = nn.Linear(h*w*parameters.final_filter_size, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
        x = self.lc_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
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
        return self.lc_layer.get_weight()
