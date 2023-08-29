import torch
import numpy as np
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

def fine_tune(model, epochs=50, clip_value=1.0, early_stop_threshold=5, device=torch.device('cuda')):
  train_loader = DataLoader(emnist_raw_augment, batch_size=16, shuffle=True, num_workers=2)
  valid_loader = DataLoader(emnist_test, batch_size=16, shuffle=True, num_workers=2)
  valid_loss_list = [np.inf]
  early_stop_counter = 0
  model.train()
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()

  for ep in tqdm(range(epochs)):
    for step, (x, y) in enumerate(train_loader):
      if torch.cuda.is_available():
        x = x.float().to(device)
        y = y.to(device)
      optimizer.zero_grad()
      p = model(x)
      loss = criterion(p, y)
      loss.backward()
      clip_grad_norm_(model.parameters(), clip_value)
      optimizer.step()

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
      for x, y in valid_loader:
        if torch.cuda.is_available():
          x = x.float().to(device)
          y = y.to(device)
        p = model(x)
        loss = criterion(p, y)
        valid_loss += loss.item()

    if valid_loss/len(valid_loader) < valid_loss_list[-1]:
      early_stop_counter = 0
    else:
      early_stop_counter += 1

    valid_loss_list.append(valid_loss/len(valid_loader))
    model.train()

    if early_stop_counter >= early_stop_threshold:
      print("\nModel training finished due to early stopping.")
      break