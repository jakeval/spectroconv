import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import local_layer, max_norm_constraint


class LcClfNorm(nn.Module):
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
        self.local_dim = parameters.local_dim

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
        
        lc = []
        if self.local_dim is not None:
            self.lc_layer = local_layer.LocallyConnectedConv(self.local_dim, (h,w) , final_num_channels, parameters.lc_channel, padding='same')
            lc.append(self.lc_layer)
        else:
            self.lc_layer = local_layer.LocallyConnected((h,w) , final_num_channels, parameters.lc_channel, padding='same')
            lc.append(self.lc_layer)
        self.max_norm_layers.append(self.lc_layer)
        lc.append(nn.BatchNorm2d(parameters.lc_channel))
        lc.append(nn.ReLU())
        if parameters.dropout_conv > 0:
            lc.append(nn.Dropout(p=parameters.dropout_conv)) # don't use spatial dropout -- filter maps are less correlated
        lc.append(self.pool)
        h = h // 2
        w = w // 2
        self.lc = nn.Sequential(*lc)

        self.fc1 = nn.Linear(h*w*parameters.lc_channel, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.max_norm_layers.append(self.fc1)
        self.max_norm_layers.append(self.fc2)
        
    def forward(self, x):
        if self.training and self.max_norm is not None:
            max_norm_constraint.max_norm(self.max_norm_layers, self.max_norm)
        x = self.convs(x)
        x = self.lc(x)
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

    def get_weights(self):
        return self.lc_layer.get_weight()


class LCTaenzer(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super(LCTaenzer, self).__init__()
        h, w = input_shape

        self.norm_input = nn.BatchNorm2d(1, affine=False)
        
        convs = []
        self.kernel_sizes = [parameters.kernel_size] * parameters.num_conv_layers
        self.channels = [1] + [parameters.num_channels * 2**i for i in range(0, parameters.num_conv_layers)]

        for i in range(len(self.kernel_sizes)):
            # i -> i+1
            convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding=2, stride=1))
            convs.append(nn.BatchNorm2d(self.channels[i+1]))
            convs.append(nn.ReLU())

            convs.append(nn.Conv2d(self.channels[i+1], self.channels[i+1], self.kernel_sizes[i], padding=2, stride=1))
            convs.append(nn.BatchNorm2d(self.channels[i+1]))
            convs.append(nn.ReLU())

            h += 4
            w += 4

            convs.append (nn.MaxPool2d(3,3))

            if parameters.dropout > 0:
              convs.append(nn.Dropout2d(p=parameters.dropout))
            h = h // 3
            w = w // 3
            #print('h:', h, 'w', w)          
        self.convs = nn.Sequential(*convs)


        # Add LC layer
        lc = []
        if parameters.local_dim is None:
            lc.append(local_layer.LocallyConnected((h, w), self.channels[-1], parameters.lc_channels, 3, 1, 2))
        else:
            lc.append(local_layer.LocallyConnectedConv(parameters.local_dim, (h, w), self.channels[-1], parameters.lc_channels, 3, 1, 2))

        lc.append(nn.BatchNorm2d(parameters.lc_channels))
        lc.append(nn.ReLU())

        if parameters.local_dim is None:
            lc.append(local_layer.LocallyConnected((h+2, w+2), parameters.lc_channels, parameters.lc_channels, 3, 1, 2))
        else:
            lc.append(local_layer.LocallyConnectedConv(parameters.local_dim, (h+2, w+2), parameters.lc_channels, parameters.lc_channels, 3, 1, 2))

        lc.append(nn.BatchNorm2d(parameters.lc_channels))
        lc.append(nn.ReLU())

        lc.append(nn.MaxPool2d(3,3))
        if parameters.dropout > 0:
            lc.append(nn.Dropout(p=parameters.dropout))
        self.lc = nn.Sequential(*lc)


        self.fc1 = nn.Linear(parameters.lc_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
        x = self.lc(x)
        x = torch.amax(x, dim=(2, 3))
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
        conv_count = 0
        linear_count = 0
        lc_count = 0
        weights = []
        names = []
        for layer in self.modules():
            w = None
            name = None
            if isinstance(layer, nn.Conv2d):
                name = f"conv{conv_count}"
                conv_count += 1
                w = layer.weight
            if isinstance(layer, nn.Linear):
                name = f"linear{linear_count}"
                linear_count += 1
                w = layer.weight
            if isinstance(layer, local_layer.LocallyConnected) or isinstance(layer, local_layer.LocallyConnectedConv):
                name = f"lc{lc_count}"
                lc_count += 1
                w = layer.get_weight()
            if w is not None:
                names.append(name)
                weights.append(w)        
        return list(zip(names, weights))

    def get_local_weights(self):        
        weights = []
        names = []
        weights.append(self.lc1.get_weight(tile=True))
        weights.append(self.lc2.get_weight(tile=True))
        names.append("lc0")
        names.append("lc1")
