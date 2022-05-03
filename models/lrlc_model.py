import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers import lrlc_layer, max_norm_constraint


class LrlcClf(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.max_norm_layers = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [parameters.kernel_size] * len(parameters.num_channels)
        self.channels = [1] + parameters.num_channels
        if parameters.dropout_fc:
            self.dropout_fc = nn.Dropout(parameters.dropout_fc)
        if parameters.dropout_input:
            self.dropout_input = nn.Dropout(parameters.dropout_input)
        self.max_norm = parameters.max_norm

        self.local_dim = parameters.local_dim
        self.rank = parameters.rank
        self.lcw = parameters.lcw # T/F
        self.lrlc_channels = parameters.lrlc_channels # 8

        for i in range(len(self.kernel_sizes)):
            self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1))
            self.max_norm_layers.append(self.convs[-1])
            self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
            self.convs.append(nn.ReLU())
            if parameters.dropout_conv > 0:
              self.convs.append(nn.Dropout2d(p=parameters.dropout_conv))
            self.convs.append(self.pool)
            h = h // 2
            w = w // 2
        #print('h', h, 'w', w)
        self.convs = nn.Sequential(*self.convs)

        lrlc = []
        self.lrlc_layer = lrlc_layer.LowRankLocallyConnected(self.rank, (h,w), self.channels[-1], self.lrlc_channels, padding='same', local_dim=self.local_dim)
        if self.lcw:
            self.combining_weights = lrlc_layer.LocalizationCombiningWeights(self.rank, (h,w), self.lrlc_layer.L, self.channels[-1], parameters.low_dim, local_dim=self.local_dim)
        else:
            self.combining_weights = lrlc_layer.OuterCombiningWeights(self.rank, self.lrlc_layer.L, local_dim=self.local_dim)
        self.max_norm_layers.append(self.lrlc_layer)
        lrlc.append(nn.BatchNorm2d(parameters.lrlc_channels))
        lrlc.append(nn.ReLU())
        if parameters.dropout_conv > 0:
            lrlc.append(nn.Dropout(p=parameters.dropout_conv)) # don't use spatial dropout -- filter maps are less correlated
        lrlc.append(self.pool)
        self.lrlc_followup = nn.Sequential(*lrlc)
        h = h // 2
        w = w // 2

        self.fc1 = nn.Linear(h*w*self.lrlc_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.max_norm_layers.append(self.fc1)
        self.max_norm_layers.append(self.fc2)

        self.class_names = class_enums

    def forward(self, x):
        if self.training and self.max_norm is not None:
            max_norm_constraint.max_norm(self.max_norm_layers, self.max_norm)
        x = self.convs(x)
        cw = None
        if self.lcw:
            cw = self.combining_weights(x)
        else:
            cw = self.combining_weights()
        x = self.lrlc_layer(x, cw)
        x = self.lrlc_followup(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
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
        return self.lrlc_layer.get_weight(self.combining_weights())


class TaenzerLRLCBlock(nn.Module):
    def __init__(self, in_shape, rank, in_channels, out_channels, local_dim=None, lcw=False, low_dim=None, dropout=0, device="cpu"):
        super(TaenzerLRLCBlock, self).__init__()
        self.lrlc1 = lrlc_layer.LowRankLocallyConnected(rank, in_shape, in_channels, out_channels, 3, stride=1, padding=2, local_dim=local_dim)
        self.lcw = lcw
        if lcw:
            self.combining_weights1 = lrlc_layer.LocalizationCombiningWeights(rank, in_shape, self.lrlc1.L, in_channels, low_dim, local_dim=local_dim, device=device)
        else:
            self.combining_weights1 = lrlc_layer.OuterCombiningWeights(rank, self.lrlc1.L, local_dim=local_dim)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.lrlc2 = lrlc_layer.LowRankLocallyConnected(rank, self.lrlc1.L, out_channels, out_channels, 3, stride=1, padding=2, local_dim=local_dim)
        if lcw:
            self.combining_weights2 = lrlc_layer.LocalizationCombiningWeights(rank, self.lrlc1.L, self.lrlc2.L, in_channels, low_dim, local_dim=local_dim, device=device)
        else:
            self.combining_weights2 = lrlc_layer.OuterCombiningWeights(rank, self.lrlc2.L, local_dim=local_dim)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(3,3)

    def forward(self, x):
        cw1 = None
        if self.lcw:
            cw1 = self.combining_weights1(x)
        else:
            cw1 = self.combining_weights1()
        x = self.lrlc1(x, cw1)
        x = F.relu(self.bn1(x))

        cw2 = None
        if self.lcw:
            cw2 = self.combining_weights2(x)
        else:
            cw2 = self.combining_weights2()
        x = self.lrlc2(x, cw2)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        return x


class LRLCTaenzer(nn.Module):
    def __init__(self, input_shape, class_enums, parameters, device="cpu"):
        super(LRLCTaenzer, self).__init__()
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

        # Add LRLC layer
        self.lrlc = TaenzerLRLCBlock((h, w), parameters.rank, self.channels[-1], parameters.lrlc_channels, local_dim=parameters.local_dim, lcw=parameters.lcw, low_dim=parameters.low_dim, dropout=parameters.dropout, device=device)
        final_num_channels = self.channels[-1]
        
        #print('CNN last layer h:', h, 'w', w, 'final channels:', final_num_channels)
        
        self.fc1 = nn.Linear(final_num_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
        x = self.lrlc(x)
        x = torch.amax(x, dim=(2, 3)) # should now be 256 (final_num_channels) in length
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
