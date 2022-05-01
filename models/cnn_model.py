import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import max_norm_constraint


class CnnClf(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [parameters.kernel_size] * len(parameters.num_channels)
        self.channels = [1] + parameters.num_channels
        self.dropout_fc, self.dropout_input = None, None
        if parameters.dropout_fc:
            self.dropout_fc = nn.Dropout(parameters.dropout_fc)
        if parameters.dropout_input:
            self.dropout_input = nn.Dropout(parameters.dropout_input)
        self.class_names = class_enums
        self.max_norm_layers = []
        self.max_norm = parameters.max_norm

        for i in range(len(self.kernel_sizes)):
            conv = nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1)
            self.convs.append(conv)
            self.max_norm_layers.append(conv)
            self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
            self.convs.append(nn.ReLU())
            if parameters.dropout_conv > 0:
                self.convs.append(nn.Dropout2d(p=parameters.dropout_conv))
            self.convs.append(self.pool)
            h = h // 2
            w = w // 2
        #print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)
        
        final_num_channels = parameters.num_channels[-1]
        self.fc1 = nn.Linear(h * w * final_num_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        
        self.max_norm_layers.append(self.fc1)
        self.max_norm_layers.append(self.fc2)

    def forward(self, x):
        if self.training and self.max_norm is not None:
            max_norm_constraint.max_norm(self.max_norm_layers, self.max_norm)
        if self.dropout_input is not None:
            x = self.dropout_input(x)
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.dropout_fc is not None:
            x = self.dropout_fc(x)
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


class OriginalCNN(nn.Module):
    # original had batch_size = 64, SGD with 0.005 lr, 0.9 momentum and 0 regularization
    # and this model below
    def __init__(self, input_shape, class_enums):
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
        #print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)
        
        #self.aap = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(self.channels[-1], num_classes)
        #self.fc1 = nn.Linear(h*w*32, 64)
        self.fc1 = nn.Linear(h*w*32, 64)
        self.fc2 = nn.Linear(64, len(class_enums))
        self.class_names = class_enums
        
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

    def loss(self, S, y, **kwargs):
        softmax = nn.CrossEntropyLoss()
        return softmax(S, y)
        #return F.cross_entropy(S, y, **kwargs)

    def ordinal_to_class_enum(self, y):
        y2 = [None] * len(y)
        y = y.squeeze()
        for i in range(len(y)):
            y2[i] = self.class_names[y[i]]
        return y2

