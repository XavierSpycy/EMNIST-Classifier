import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple

class ImageDataset(Dataset):
  def __init__(self, images, labels, mean: float=0.1736, std: float=0.3248, is_augment: bool=False) -> None:
    """
    Initialize ImageDataset class

    Parameters:
    - images (np.ndarray): array of images.
    - labels (np.ndarray): array of labels.
    - mean (float): mean of images. Default: 0.1736.
    - std (float): standard deviation of images. Default: 0.3248.
    - is_augment (bool): whether to augment images or not. Default: False.
    """
    self.images = images
    self.labels = labels
    # If is_augment is True, then augment images
    if is_augment:
      self.transform = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize((mean, ), (std, ))])
    else:
      self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((mean,), (std,))])

  def __len__(self):
    """
    Override __len__ method
    """
    return len(self.images)

  def __getitem__(self, idx):
    """
    Override __getitem__ method
    """
    img = self.images[idx]
    lab = self.labels[idx]
    img = self.transform(img)
    return img, lab
  
def load_data_from_file(train_file: str='data/emnist_train.pkl', 
                        test_file: str='data/emnist_test.pkl') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Load data from file.

  Parameters:
  - train_file (str): path to training data file. Default: 'data/emnist_train.pkl'.
  - test_file (str): path to testing data file. Default: 'data/emnist_test.pkl'.

  Returns:
  - X_train (np.ndarray): array of training images.
  - y_train (np.ndarray): array of training labels.
  - X_test (np.ndarray): array of testing images.
  - y_test (np.ndarray): array of testing labels.
  """
  with open(train_file, 'rb') as train:
    train_set = pickle.load(train)
  with open(test_file, 'rb') as test:
    test_set = pickle.load(test)
  X_train = train_set['data']
  y_train = train_set['labels']
  X_test = test_set['data']
  y_test = test_set['labels']
  return X_train, y_train, X_test, y_test

def split_data(X, y, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Split data into training set, validation set, and mini set.

  Parameters:
  - X (np.ndarray): array of images.
  - y (np.ndarray): array of labels.
  - random_state (int): random state. Default: 42.

  Returns:
  - X_train (np.ndarray): array of training images.
  - X_mini (np.ndarray): array of mini images.
  - y_train (np.ndarray): array of training labels.
  - y_mini (np.ndarray): array of mini labels.
  - X_mini_train (np.ndarray): array of mini training images.
  - X_mini_val (np.ndarray): array of mini validation images.
  - y_mini_train (np.ndarray): array of mini training labels.
  - y_mini_val (np.ndarray): array of mini validation labels.
  """
  X_train, X_mini, y_train, y_mini = train_test_split(X, y, test_size=0.1,
                                                      random_state=random_state, stratify=y)
  X_mini_train, X_mini_val, y_mini_train, y_mini_val = train_test_split(X_mini, y_mini, test_size=0.2, 
                                                                        random_state=random_state, stratify=y_mini)
  return X_train, X_mini, y_train, y_mini, X_mini_train, X_mini_val, y_mini_train, y_mini_val

def load_dataset(mode='train') -> Tuple[ImageDataset, ImageDataset, ImageDataset, ImageDataset, ImageDataset, ImageDataset, ImageDataset, ImageDataset, ImageDataset]:
  """
  Load dataset.

  Parameters:
  - mode (str): mode to load dataset. Default: 'train'.

  Returns:
  - emnist_raw (ImageDataset): raw EMNIST dataset.
  - emnist_train (ImageDataset): EMNIST training dataset.
  - emnist_test (ImageDataset): EMNIST testing dataset.
  - emnist_mini (ImageDataset): EMNIST mini dataset.
  - emnist_mini_train (ImageDataset): EMNIST mini training dataset.
  - emnist_mini_val (ImageDataset): EMNIST mini validation dataset.
  - emnist_raw_augment (ImageDataset): raw EMNIST dataset with augmentation.
  - emnist_augment (ImageDataset): EMNIST training dataset with augmentation.
  - emnist_augment_mini (ImageDataset): EMNIST mini training dataset with augmentation.
  """
  X_train_raw, y_train_raw, X_test, y_test = load_data_from_file()
  X_train, X_mini, y_train, y_mini, X_mini_train, X_mini_val, y_mini_train, y_mini_val = split_data(X_train_raw, y_train_raw)
  # Convert ndarray to PIL Image
  image_train_raw = [Image.fromarray(ndarray, mode='L') for ndarray in X_train_raw]
  image_train = [Image.fromarray(ndarray, mode='L') for ndarray in X_train]
  image_test = [Image.fromarray(ndarray, mode='L') for ndarray in X_test]
  image_mini = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini]
  image_mini_train = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini_train]
  image_mini_val = [Image.fromarray(ndarray, mode='L') for ndarray in X_mini_val]
  # Convert ndarray to tensor
  y_train_raw_tensor = torch.from_numpy(y_train_raw)
  y_train_tensor = torch.from_numpy(y_train)
  y_mini_tensor = torch.from_numpy(y_mini)
  y_test_tensor = torch.from_numpy(y_test)
  y_mini_train_tensor = torch.from_numpy(y_mini_train)
  y_mini_val_tensor = torch.from_numpy(y_mini_val)
  # Encapsulate data into ImageDataset class
  emnist_raw = ImageDataset(image_train_raw, y_train_raw_tensor)
  emnist_train = ImageDataset(image_train, y_train_tensor)
  emnist_test = ImageDataset(image_test, y_test_tensor)
  emnist_mini = ImageDataset(image_mini, y_mini_tensor)
  emnist_mini_train = ImageDataset(image_mini_train, y_mini_train_tensor)
  emnist_mini_val = ImageDataset(image_mini_val, y_mini_val_tensor)
  emnist_raw_augment = ImageDataset(image_train_raw, y_train_raw_tensor, is_augment=True)
  emnist_augment = ImageDataset(image_train, y_train_tensor, is_augment=True)
  emnist_augment_mini = ImageDataset(image_mini_train, y_mini_train_tensor, is_augment=True)
  if mode == 'train':
    return emnist_raw, emnist_train, emnist_test, emnist_mini, emnist_mini_train, emnist_mini_val, emnist_raw_augment, emnist_augment, emnist_augment_mini
  elif mode == 'eval':
    return emnist_raw, emnist_test