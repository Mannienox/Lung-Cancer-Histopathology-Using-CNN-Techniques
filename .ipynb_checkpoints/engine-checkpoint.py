from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import pandas as pd
from torch import nn
import torch
import numpy as np

def train_step(model, loss_fn, optimizer, loader, device):

    # Initializing lists to keep records
    train_acc, train_loss = 0,0
    model.train()

    for batch, (X,y) in enumerate(loader):
        X,y = X.to(device), y.to(device)

        # Train
        y_pred_logits = model(X)

        #Train Prediction
        y_pred = torch.softmax(y_pred_logits, dim = 1).argmax(dim = 1)

        #
        loss = loss_fn(y_pred_logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Computing accuracy
        acc = accuracy_score(y.cpu(), y_pred.cpu())
        train_loss += loss.item()
        train_acc += acc

    return train_acc/len(loader) * 100, train_loss/len(loader)

def test_step(model, loss_fn, optimizer, loader, device):


    # Initializing lists to keep records
    test_acc, test_loss = 0,0

    model.eval()

    for batch, (X,y) in enumerate(loader):
        with torch.inference_mode():
            X,y = X.to(device), y.to(device)
            # Prediction
            test_pred_logits = model(X)

            #Train Prediction
            test_pred = torch.softmax(test_pred_logits, dim = -1).argmax(dim = -1)
            loss = loss_fn(test_pred_logits,y)

            # Computing accuracy
            acc = accuracy_score(y.cpu(), test_pred.cpu())
            test_loss += loss.item()
            test_acc += acc

    return test_acc/len(loader) * 100, test_loss/len(loader)

def train_loop (
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        loss_fn,
        epochs):

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in tqdm(range(epochs)):
        model.to(device)

        # Training
        train_acc, train_loss = train_step(model, loss_fn, optimizer, train_loader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Testing
        test_acc, test_loss = test_step(model, loss_fn, optimizer, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    results = pd.DataFrame({
        'Epoch': np.arange(1, epochs+1),
        'Train Accuracy': train_accs,
        'Train Loss' : train_losses,
        'Test Accuracy': test_accs,
        'Test Loss' : test_losses,
    })

    return results
