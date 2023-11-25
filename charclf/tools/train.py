import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler

def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, num_epochs):
  '''Compute the validation loss during the training epochs'''
  """Parameters:
      model, corresponds to a specific classifier;
      optimizer, corresponds to a PyTorch optimizer, typically SGD and Adam;
      criterion, corresponds to Cross Entropy Loss in our context;
      train_loader, corresponds to a PyTorch DataLoader containing training inputs and labels;
      valid_loader, corresponds to a PyTorch DataLoader containing validation inputs and labels;
      num_epochs, correspond to the number of training epochs.

      Return:
      val_loss, the loss value on the validation set durining the specific training procedure.
  """
  # Switch the model to the training state
  model.train()
  # Train model for several epochs
  for epoch in range(num_epochs):
    for inputs, targets in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

  # Switch the model to the evaluation state
  model.eval()
  # Compute the loss value on the validation set
  val_loss = 0
  with torch.no_grad():
    for inputs, targets in val_loader:
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      val_loss += loss.item()

  val_loss /= len(val_loader)
  return val_loss

def random_search(model_class, param_grid, train_loader_tuning, val_loader_tuning, n_iter=28, num_epochs=5):
  """Random search for hyperparameter tuning.

  Parameters:
  - model_class: a specific classifier.
  - param_grid: a dictionary containing the hyperparameter search space.
  - n_iter (int): the number of iterations for random search.
  - num_epochs (int): the number of training epochs.

  Return:
  - best_params (dict): the best hyperparameters.
  """
  param_list = list(ParameterSampler(param_grid, n_iter))
  best_loss = float('inf')
  best_params = None
  criterion = nn.CrossEntropyLoss()
  # Use a Python set to aviod repeated combinations
  params_attempt = set()

  for params in tqdm(param_list):
    params_tuple = tuple(params.items())
    if params_tuple in params_attempt:
      continue
    else:
      params_attempt.add(params_tuple)

    # Initialise a model
    model = model_class()
    # Optimizer containing 3 hyperparameters
    optimizer = optim.SGD(model.parameters(),
                          lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    # Compute the validation loss value based on the function we defined before
    val_loss = train_and_evaluate(model, optimizer, criterion, train_loader_tuning, val_loader_tuning, num_epochs)

    # Compare the loss value
    if val_loss < best_loss:
      best_loss = val_loss
      best_params = params

  return best_params

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

def train_(model, train_loader, optimizer, criterion, epochs, device=torch.device('cuda')):
  for ep in tqdm(range(epochs)):
    for step, (x, y) in enumerate(train_loader):
      x = x.float().to(device)
      y = y.to(device)
      optimizer.zero_grad()
      p = model(x)
      loss = criterion(p, y)
      loss.backward()
      optimizer.step()