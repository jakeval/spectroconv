import torch
from torch import nn
import numpy as np
from torch import optim
from models import cnn_model


def train_model_dataloader(clf: cnn_model.CnnClf, dataloader, dataloader_val, lr=0.005, momentum=0.9, epochs=80):
    softmax = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum)
    val_iter = iter(dataloader_val)

    print("Start Training...")
    losses = []
    train_accuracies = []
    val_accuracies = []
    print_increments = int(epochs // 10)
    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = clf(inputs)
            loss = softmax(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_accuracy += (clf.predict_from_scores(outputs.detach().numpy()) == labels.detach().numpy()).mean()
        epoch_loss = running_loss / (i+1)
        epoch_accuracy = running_accuracy / (i+1)
        with torch.no_grad():
            train_accuracies.append(epoch_accuracy)
            val_data = next(val_iter, None)
            if val_data is None:
                val_iter = iter(dataloader_val)
                val_data = next(dataloader_val)
            val_X, val_y = val_data
            val_accuracies.append(get_accuracy(clf, val_X.numpy(), val_y.numpy()))
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies


def train_model(clf: cnn_model.CnnClf, X, y, X_val, y_val, lr=0.005, momentum=0.9, epochs=80):
    X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
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
            train_accuracies.append(get_accuracy(clf, X, y, size=128))
            val_accuracies.append(get_accuracy(clf, X_val, y_val, size=128))
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies


def get_accuracy(clf: cnn_model.CnnClf, X, y, size=None):
    if size is not None:
        idx = np.random.choice(np.arange(X.shape[0]), size=size, replace=False)
        return (clf.predict(X[idx]) == y[idx]).mean()
    else:
        return (clf.predict(X) == y).mean()


def stream_accuracy(clf, dataloader):
    accuracies = []
    batch_sizes = []
    for data in dataloader:
        X, y = data
        accuracies.append(get_accuracy(clf, X.numpy(), y.numpy()))
        batch_sizes.append(X.shape[0])
    accuracies = np.array(accuracies)
    batch_sizes = np.array(batch_sizes)
    weights = batch_sizes / batch_sizes.sum()
    return accuracies @ weights