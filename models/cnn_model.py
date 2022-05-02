import torch
from torch import nn
from torch.nn import functional as F
import numpy as np



class CnnClf(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        
        self.convs = []
        self.pool = nn.MaxPool2d(2,2)
        self.kernel_sizes = [parameters.kernel_size] * parameters.num_conv_layers
        self.channels = [1] + [i * parameters.num_channels for i in range(1, parameters.num_conv_layers+1)]

        for i in range(len(self.kernel_sizes)):
            self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], padding='same', stride=1))
            self.convs.append(nn.BatchNorm2d(self.channels[i+1]))
            self.convs.append(nn.ReLU())
            if parameters.dropout > 0:
              self.convs.append(nn.Dropout2d(p=parameters.dropout))
            self.convs.append(self.pool)
            h = h // 2
            w = w // 2
        self.convs = nn.Sequential(*self.convs)
        
        #self.aap = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(self.channels[-1], num_classes)
        #self.fc1 = nn.Linear(h*w*32, 64)
        final_num_channels = parameters.num_channels * parameters.num_conv_layers
        
        print('CNN last layer h:', h, 'w', w, 'channels:', self.channels[-1], '=', h * w * final_num_channels)
        
        self.fc1 = nn.Linear(h * w * final_num_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
        #x = self.aap(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)
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





class CnnTaenzer(nn.Module):
    def __init__(self, input_shape, class_enums, parameters):
        super().__init__()
        h, w = input_shape
        print('input shape', input_shape)

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

            #h += 4
            #w += 4
            
            convs.append (nn.MaxPool2d(3,3))

            if parameters.dropout > 0:
              convs.append(nn.Dropout2d(p=parameters.dropout))
            #h = h // 3
            #w = w // 3
            #print('h:', h, 'w', w)
          
        self.convs = nn.Sequential(*convs)
        self.convs2 = convs
        
        final_num_channels = self.channels[-1]
        
        #print('CNN last layer h:', h, 'w', w, 'final channels:', final_num_channels)
        
        self.fc1 = nn.Linear(final_num_channels, parameters.linear_layer_size)
        self.fc2 = nn.Linear(parameters.linear_layer_size, len(class_enums))
        self.class_names = class_enums
        
    def forward(self, x):
        x = self.convs(x)
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
