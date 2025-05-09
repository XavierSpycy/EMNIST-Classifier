import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def show_batch(dataset: Dataset, rows:int=10, cols: int=10) -> None:
    """
    Show a batch of images.

    Parameters:
    - dataset (torch.utils.data.Dataset): dataset to show.
    - rows (int): number of rows. Default: 10.
    """
    num_display = rows * cols
    label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    indices = torch.randperm(len(dataset))[:num_display]
    _, axes = plt.subplots(rows, cols, figsize=(rows, cols))
    for idx, ax in zip(indices, axes.flatten()):
        image, label = dataset[idx]
        image_np = image.squeeze(0).numpy()
        ax.imshow(image_np, cmap='gray')
        ax.set_title(f'Label: {label_map[label]}', fontsize=8, y=.9)
        ax.axis('off')
    plt.show()

def predict(model, data_loader, device=torch.device('cuda'), num_display=100) -> None:
    """
    Visualize model predictions.

    Parameters:
    - model (torch.nn.Module): model to visualize.
    - data_loader (torch.utils.data.DataLoader): data loader to visualize on.
    - device (torch.device): device to run model on. Default: torch.device('cuda').
    - num_display (int): number of images to display. Default: 100.
    """
    model.eval()
    rows = int(np.sqrt(num_display))
    cols = rows
    all_preds = []
    all_labels = []
    counter = 0
    label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    fig, axes = plt.subplots(rows, cols, figsize=(rows, cols))

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.float().to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            counter += 1

            for i, ax in enumerate(axes.flatten()):
                if i < len(images):
                    image_np = images[i].squeeze(0).cpu().numpy()
                    pred_label = predicted[i].item()
                    true_label = labels[i].item()
                    ax.imshow(image_np, cmap='gray')
                    title = f'T: {label_map[true_label]}, P: {label_map[pred_label]}'
                    if true_label != pred_label:
                        ax.set_title(title, fontsize=8, y=.9, color='red')
                    else:
                        ax.set_title(title, fontsize=8, y=.9, color='black')
                    ax.axis('off')

            if counter == num_display:
                break
    plt.show()