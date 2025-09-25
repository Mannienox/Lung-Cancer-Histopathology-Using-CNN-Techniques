from typing import List, Dict, Tuple
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split


def prepare_data(data_path,
                 BATCH_SIZE = 32,
                 data_transforms = None,
                 random_state = None):


    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)



    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.Resize(size = (64,64)),
            #transforms.RandomRotation(degrees = 90),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
            ])

    dataset = ImageFolder(
        root = data_path,
        transform = data_transforms
    )


    # Classes
    classes = dataset.classes

    # Splitting
    train_data, test_data = train_test_split(dataset, test_size = .3, stratify = dataset.targets, random_state = random_state)

    # Loaders
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)   # So our model is in the same order each time we want to rewrite test our model

    return train_loader, test_loader, classes
