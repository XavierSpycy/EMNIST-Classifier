import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model, data_loader, device=torch.device('cuda')):
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

def confusion_matrix_viz(cm):
  label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  plt.figure(figsize=(40, 40))
  sns.set(font_scale=1.5)
  sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_map, yticklabels=label_map)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()

def multi_evaluate(models, data_loader, device=torch.device('cuda')):
  results = []
  for model in models:
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
  return results