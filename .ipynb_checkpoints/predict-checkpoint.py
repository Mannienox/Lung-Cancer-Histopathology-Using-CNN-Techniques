
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch


def predict_single (model, image, device, transform = None, labels = None):


    original_image = Image.open(image)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.ToTensor()
        ])

    img = transform(original_image)

    model.eval()
    with torch.inference_mode():
        pred_logits = model(img.unsqueeze(dim = 0))

        #Train Prediction
        pred = torch.softmax(pred_logits, dim = -1).argmax(dim = -1)
        pred_prob = torch.softmax(pred_logits, dim = -1).max().item()

        if labels:
            pred = labels[pred.item()]

        return pred, pred_prob
