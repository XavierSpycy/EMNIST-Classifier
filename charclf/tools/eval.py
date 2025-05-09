import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from charclf.models import LeNet, VGGNet, AlexNet, SpinalNet, ResNet, EnsNet
from typing import Tuple

def evaluate(model, data_loader, device=torch.device('cuda')) -> Tuple[float, float, float, float, np.ndarray]:
  """
  Evaluate model on data_loader.

  Parameters:
  - model (torch.nn.Module): model to evaluate.
  - data_loader (torch.utils.data.DataLoader): data loader to evaluate on.
  - device (torch.device): device to run model on. Default: torch.device('cuda').

  Returns:
  - acc (float): accuracy score.
  - precision (float): precision score.
  - recall (float): recall score.
  - f1 (float): f1 score.
  - cm (np.ndarray): confusion matrix.
  """
  model.eval()
  all_preds = []
  all_labels = []
  with torch.no_grad():
    for images, labels in data_loader:
      images, labels = images.float().to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
  all_preds = np.array(all_preds)
  all_labels = np.array(all_labels)
  acc = accuracy_score(all_labels, all_preds)
  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')
  f1 = f1_score(all_labels, all_preds, average='weighted')
  cm = confusion_matrix(all_labels, all_preds)
  model.train()
  return acc, precision, recall, f1, cm

def confusion_matrix_viz(cm: np.ndarray) -> None:
  """
  Visualize confusion matrix.

  Parameters:
  - cm (np.ndarray): confusion matrix.
  """
  label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  plt.figure(figsize=(40, 40))
  sns.set(font_scale=1.5)
  sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_map, yticklabels=label_map)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()

def multi_evaluate(models: list, data_loader: torch.utils.data.DataLoader, device=torch.device('cuda')) -> pd.DataFrame:
  """
  Evaluate multiple models on data_loader.

  Parameters:
  - models (list): list of models to evaluate.
  - data_loader (torch.utils.data.DataLoader): data loader to evaluate on.
  - device (torch.device): device to run model on. Default: torch.device('cuda').

  Returns:
  - df (pd.DataFrame): dataframe of evaluation results.
  """
  results = []
  index = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
  columns = []
  for model in models:
    if isinstance(model, LeNet):
      columns.append('LeNet')
    elif isinstance(model, VGGNet):
      columns.append('VGGNet')
    elif isinstance(model, AlexNet):
      columns.append('AlexNet')
    elif isinstance(model, SpinalNet):
      columns.append('SpinalNet')
    elif isinstance(model, ResNet):
      columns.append('ResNet')
    elif isinstance(model, EnsNet):
      columns.append('EnsNet')
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
      for images, labels in data_loader:
        images, labels = images.float().to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    model.train()
    results.append([acc, precision, recall, f1, cm])
  results_ = [results[i][:-1] for i in range(len(results))]
  results_ = list(map(list, zip(*results_)))
  df = pd.DataFrame(results_, index=index, columns=columns)
  df = df.round(4)
  df = df.applymap(lambda x: f'{x * 100:.2f}%')
  return df