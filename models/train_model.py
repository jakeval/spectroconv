import torch
from torch import nn
import numpy as np
from torch import optim
from models import cnn_model
from sklearn.metrics import f1_score
import wandb
from tqdm.auto import tqdm
from data_utils import nsynth_adapter as na
from models import cnn_model
import json
import os
import pickle


def train_model_dataloader(clf: cnn_model.CnnClf, dataloader, dataloader_val, lr=0.005, momentum=0.9, epochs=80, device = None):
    softmax = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum)
    val_iter = iter(dataloader_val)

    print("Start Training...")
    losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1s = []
    print_increments = int(epochs // 10)
    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if device:
              inputs = inputs.to(device)
              labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = clf(inputs)
            loss = softmax(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            detached_outputs = outputs.detach()
            detached_labels = labels.detach()
            if device:
              detached_outputs = detached_outputs.cpu()
              detached_labels = detached_labels.cpu()
            running_accuracy += (clf.predict_from_scores(detached_outputs.numpy()) == detached_labels.numpy()).mean()
        
        epoch_loss = running_loss / (i+1)
        epoch_accuracy = running_accuracy / (i+1)
        with torch.no_grad():
            train_accuracies.append(epoch_accuracy)
            val_data = next(val_iter, None)
            if val_data is None:
                val_iter = iter(dataloader_val)
                val_data = next(dataloader_val)
            val_X, val_y = val_data

            detached_outputs = outputs.detach()
            detached_labels = labels.detach()
            if device:
              detached_outputs = detached_outputs.cpu()
              detached_labels = detached_labels.cpu()

            running_accuracy += (clf.predict_from_scores(detached_outputs.numpy()) == detached_labels.numpy()).mean()
            
            val_accuracy = stream_accuracy(clf, dataloader_val, device)
            val_accuracies.append(val_accuracy)
            val_f1 = stream_f1(clf, dataloader_val, device)
            val_f1s.append(val_f1)
        
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f} \t val f1: {val_f1s[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies, val_f1s


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


def get_accuracy(clf: cnn_model.CnnClf, X, y, size=None, device = None):
    if size is not None:
        idx = np.random.choice(np.arange(X.shape[0]), size=size, replace=False)
        return (clf.predict(X[idx], device) == y[idx]).mean()
    else:
        return (clf.predict(X, device) == y).mean()

def get_f1(clf: cnn_model.CnnClf, X, y, size=None, device = None):
  predictions = clf.predict(X, device)
  #print(f1_score(y, predictions, average = None))
  #print(f1_score(y, predictions, average = "macro"))
  return f1_score(y, predictions, average = "macro")



def stream_accuracy(clf, dataloader, device):
    accuracies = []
    batch_sizes = []
    for data in dataloader:
        X, y = data
        accuracies.append(get_accuracy(clf, X.numpy(), y.numpy(), device = device))
        batch_sizes.append(X.shape[0])
    accuracies = np.array(accuracies)
    batch_sizes = np.array(batch_sizes)
    weights = batch_sizes / batch_sizes.sum()
    return accuracies @ weights


def stream_f1(clf, dataloader, device):
    f1s = []
    batch_sizes = []
    for data in dataloader:
        X, y = data
        f1s.append(get_f1(clf, X.numpy(), y.numpy(), device = device))
        batch_sizes.append(X.shape[0])
    f1s = np.array(f1s)
    batch_sizes = np.array(batch_sizes)
    weights = batch_sizes / batch_sizes.sum()
    return f1s @ weights