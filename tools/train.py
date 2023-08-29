import time
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def hr_min_sec(second):
  if second > 3600:
    hr = int(second // 3600)
    min = int((second - hr * 3600) // 60)
    sec = second % 60
    print(f"Training procedure running time: {hr} hour(s), {min} minute(s), and {sec:.2f} second(s).")
  elif second > 60:
    min = int(second // 60)
    sec = second % 60
    print(f"Training procedure running time: {min} minute(s), and {sec:.2f} second(s).")
  else:
    sec = second % 60
    print(f"Training procedure running time: {sec:.2f} second(s).")

def topk_accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0/batch_size))
  return res

def accuracy(output, target):
  batch_size = target.size(0)
  _, pred = output.topk(1, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  correct_1 = correct[:1].view(-1).float().sum(0)
  res = correct_1.mul_(100.0 / batch_size)
  return res

def train(model, train_loader, valid_loader, optimizer, criterion, epochs=20, clip_value=1.0, early_stop_threshold=10, device=torch.device('cuda')):
  loss_list = []
  acc_list = []
  valid_loss_list = [np.inf] 
  valid_acc_list = []
  early_stop_counter = 0
  model.train()
  start_time = time.time()
  for ep in tqdm(range(epochs)):
    ep_loss = 0.0
    ep_acc = 0.0
    for step, (x, y) in enumerate(train_loader):
      if torch.cuda.is_available():
        x = x.float().to(device)
        y = y.to(device)
      optimizer.zero_grad()
      p = model(x)
      loss = criterion(p, y)
      acc = accuracy(p, y)
      ep_loss += loss.item()
      ep_acc += acc.item()
      loss.backward()
      clip_grad_norm_(model.parameters(), clip_value)
      optimizer.step()
    loss_list.append(ep_loss/(step+1))
    acc_list.append(ep_acc/(step+1))

    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
      for x, y in valid_loader:
        if torch.cuda.is_available():
          x = x.float().to(device)
          y = y.to(device)
        p = model(x)
        loss = criterion(p, y)
        acc = accuracy(p, y)
        valid_loss += loss.item()
        valid_acc += acc.item()
    if valid_loss/len(valid_loader) < valid_loss_list[-1]:
      early_stop_counter = 0
    else:
      early_stop_counter += 1

    valid_loss_list.append(valid_loss/len(valid_loader))
    valid_acc_list.append(valid_acc/len(valid_loader))
    model.train()
    if early_stop_counter >= early_stop_threshold:
      print("\nModel training finished due to early stopping.")
      break

  end_time = time.time()
  training_time = end_time - start_time
  print()
  hr_min_sec(training_time)
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(loss_list, color='red', label='Training Loss')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[0].set_title('Loss during Training')
  axs[0].legend(loc='best')
  axs[1].plot(acc_list, color=(1.0, 0.5, 0.0), label='Training Accuracy')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Accuracy')
  axs[1].set_title('Accuracy during Training')
  axs[1].legend(loc='best')
  plt.show()
     