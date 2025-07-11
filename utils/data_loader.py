import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

class CustomDataset(Dataset):
  """
  Custom Dataset for Fashion-MNIST images.
  Loads image data from a CSV and applies transformations.
  """
  def __init__(self, features, labels, transform=None):
    """
    Initializes the CustomDataset.

    Args:
      features (np.ndarray): Image pixel data (flattened).
      labels (np.ndarray): Image labels.
      transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
    """
    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)
    self.transform = transform

  def __len__(self):
    """
    Returns the total number of samples in the dataset.
    """
    return len(self.features)

  def __getitem__(self, idx):
    """
    Retrieves a sample and its label at the given index.

    Args:
      idx (int): Index of the sample to retrieve.

    Returns:
      tuple: A tuple containing the transformed image and its label.
    """
    label = self.labels[idx]
    sample = self.features[idx]

    # Reshape the flattened image to 28x28 and convert to PIL Image for transformations
    image = sample.reshape(28, 28)
    # Convert to uint8 for PIL Image.fromarray
    image = image.numpy().astype(np.uint8)
    image = Image.fromarray(image)

    if self.transform:
        image = self.transform(image)

    return image, label


def get_fashion_mnist_dataloader(data_path, batch_size=128):
    """
    Loads the Fashion-MNIST dataset from a CSV, splits it, and returns a DataLoader.

    Args:
      data_path (str): Path to the fashion-mnist_train.csv file.
      batch_size (int): Batch size for the DataLoader.

    Returns:
      torch.utils.data.DataLoader: DataLoader for the training set.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download 'fashion-mnist_train.csv' and place it there.")

    df_train = pd.read_csv(data_path)

    X = df_train.iloc[:, 1:].values # Pixel data
    y = df_train.iloc[:, 0].values  # Labels

    # Transformations: Convert to Tensor and Normalize to [-1, 1]
    # GANs often perform better with input images normalized to [-1, 1]
    custom_transforms = transforms.Compose([
        transforms.Resize(28), 
        transforms.ToTensor(), # Converts PIL Image to FloatTensor and scales to [0, 1]
        transforms.Normalize((0.5,), (0.5,)) # Normalizes to [-1, 1]
    ])

    train_dataset = CustomDataset(X, y, custom_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader

