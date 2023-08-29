import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_visualise(model, data_loader, device=torch.device('cuda'), num_display=100):
  model.eval()
  rows = int(np.sqrt(num_display))
  cols = rows
  all_preds = []
  all_labels = []
  counter = 0
  label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

  fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

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