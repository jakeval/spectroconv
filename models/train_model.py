import torch
from torch import nn
import numpy as np
from torch import optim
from models import cnn_model


def train_model(clf: cnn_model.CnnClf, X, y, X_val, y_val, lr=0.005, momentum=0.9, epochs=80):
    X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
    y = id_to_ordinal(y)
    y_val = id_to_ordinal(y_val)
    trainloader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=min(64, X.shape[0]), shuffle=True)
    softmax = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum)
    
    print("Check starting accuracy...")
    print(get_accuracy(clf, X, y), get_accuracy(clf, X_val, y_val))

    print("Start Training...")
    losses = []
    train_accuracies = []
    val_accuracies = []
    print_increments = int(epochs // 10)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = clf(inputs)
            loss = softmax(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss / (i+1)
        with torch.no_grad():
            train_accuracies.append(get_accuracy(clf, X, y))
            val_accuracies.append(get_accuracy(clf, X_val, y_val))
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies


def id_to_ordinal(y):
    codes = np.unique(y)
    y_encoded = y.copy()
    lookup = dict([(code, i) for i, code in enumerate(codes)])
    for code in codes:
        y_encoded[y == code] = lookup[code]
    return y_encoded


def get_accuracy(clf: cnn_model.CnnClf, X, y):
    idx = np.random.choice(np.arange(X.shape[0]), size=128, replace=False)
    return (clf.predict(X[idx]) == y[idx]).mean()
