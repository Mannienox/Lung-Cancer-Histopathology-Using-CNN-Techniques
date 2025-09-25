from typing import List, Dict, Tuple
from plot_lib import plot_loss_curves
import torch
from torch import nn
import engine
from sklearn.metrics import classification_report

def train_efficientnet(model,
                weights,
                in_features,
                train_loader,
                test_loader,
                classes,
                device,
                optimizer = 'sgd',
                lr = 0.1,
                loss_fn = nn.CrossEntropyLoss(),
                BATCH_SIZE = 32,
                data_transforms = None,
                epochs = 10,
                random_state = 42):


    # Freezing features
    for param in model.features.parameters():
        param.requires_grad = False

    # Adjusting classifier
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),       # First linear layer to a hidden size of 512
        nn.ReLU(),                         # Non-linear activation function
        nn.Dropout(p=0.5),                 # Dropout for regularization to prevent overfitting
        nn.Linear(512, len(classes))        # Final linear layer to output the number of classes
    )

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Training
    results = engine.train_loop(
        model = model,
        loss_fn = loss_fn,
        optimizer = torch.optim.Adam(params = model.parameters(), lr = lr) if optimizer.lower() == 'adam' else torch.optim.SGD(params = model.parameters(), lr = lr),
        train_loader = train_loader,
        test_loader = test_loader,
        epochs = epochs,
        device = device
    )


    fig = plot_loss_curves(epochs=results["Epoch"],
                     train_loss=results["Train Loss"],
                     test_loss= results["Test Loss"],
                     train_acc=results["Train Accuracy"],
                     test_acc= results["Test Accuracy"])

    #report = classification_report(resul)

    return results, fig#, report

def train_resnet(model,
                weights,
                in_features,
                train_loader,
                test_loader,
                classes,
                device,
                optimizer = 'adam',
                lr = 0.1,
                loss_fn = nn.CrossEntropyLoss(),
                BATCH_SIZE = 32,
                data_transforms = None,
                epochs = 10,
                random_state = 42):


    # Freezing features
    for param in model.parameters():
        param.requires_grad = False

    # Adjusting classifier
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),       # First linear layer to a hidden size of 512
        nn.ReLU(),                         # Non-linear activation function
        nn.Dropout(p=0.2),                 # Dropout for regularization to prevent overfitting
        nn.Linear(512, len(classes))        # Final linear layer to output the number of classes
        )
    
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Training
    results = engine.train_loop(
        model = model,
        loss_fn = loss_fn,
        optimizer = torch.optim.Adam(params = model.parameters(), lr = lr) if optimizer.lower() == 'adam' else torch.optim.SGD(params = model.parameters(), lr = lr),
        train_loader = train_loader,
        test_loader = test_loader,
        epochs = epochs,
        device = device
    )


    fig = plot_loss_curves(epochs=results["Epoch"],
                     train_loss=results["Train Loss"],
                     test_loss= results["Test Loss"],
                     train_acc=results["Train Accuracy"],
                     test_acc= results["Test Accuracy"])

    return results, fig
