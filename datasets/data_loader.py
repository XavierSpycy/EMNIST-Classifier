import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

mean, std = 0.1736, 0.3248
train_file = 'EMNISTByClassRecognition/data/emnist_train.pkl'
test_file = 'EMNISTByClassRecognition/data/emnist_test.pkl'

with open(train_file, 'rb') as train:
  train_set = pickle.load(train)
with open(test_file, 'rb') as test:
  test_set = pickle.load(test)

X_train_raw = train_set['data']
y_train_raw = train_set['labels']
X_test = test_set['data']
y_test = test_set['labels']

def mini_set(X, y, test_size=0.1, random_state=42):
  X_train, X_mini, y_train, y_mini = train_test_split(X, y, test_size=test_size,
                                                      random_state=random_state, stratify=y)
  return X_train, X_mini, y_train, y_mini

X_train, X_mini, y_train, y_mini = mini_set(X_train_raw, y_train_raw)
X_mini_train, X_mini_val, y_mini_train, y_mini_val = train_test_split(X_mini, y_mini, test_size=0.2, stratify=y_mini)

y_train_raw_tensor = torch.from_numpy(y_train_raw)
y_train_tensor = torch.from_numpy(y_train)
y_mini_tensor = torch.from_numpy(y_mini)
y_test_tensor = torch.from_numpy(y_test)
y_mini_train_tensor = torch.from_numpy(y_mini_train)
y_mini_val_tensor = torch.from_numpy(y_mini_val)

class ImageDataset(Dataset):
  def __init__(self, images, labels):
    self.images = images
    self.labels = labels
    self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,), (std,))])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = self.images[idx]
    lab = self.labels[idx]
    img = self.transform(img)
    return img, lab

image_train_raw = [Image.fromarray(ndarray, mode='L') for ndarray in X_train_raw]
image_train = [Image.fromarray(ndarray, mode='L') for ndarray in X_train]
image_test = [Image.fromarray(ndarray, mode='L') for ndarray in X_test]
image_mini = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini]
image_mini_train = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini_train]
image_mini_val = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini_val]

emnist_raw = ImageDataset(image_train_raw, y_train_raw_tensor)
emnist_train = ImageDataset(image_train, y_train_tensor)
emnist_test = ImageDataset(image_test, y_test_tensor)
emnist_mini = ImageDataset(image_mini, y_mini_tensor)
emnist_mini_train = ImageDataset(image_mini_train, y_mini_train_tensor)
emnist_mini_val = ImageDataset(image_mini_val, y_mini_val_tensor)

class AugmentDataset(Dataset):
  def __init__(self, images, labels, augment):
    self.images = images
    self.labels = labels
    # Define the data augmentation of images
    self.transform = augment

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = self.images[idx]
    lab = self.labels[idx]
    if self.transform:
      img = self.transform(img)
    return img, lab

augment = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
    transforms.ToTensor(),
    transforms.Normalize((mean, ), (std, ))
    ])

emnist_raw_augment = AugmentDataset(image_train_raw, y_train_raw_tensor, augment)
emnist_augment = AugmentDataset(image_train, y_train_tensor, augment)
emnist_augment_mini = AugmentDataset(image_mini, y_mini_tensor, augment)