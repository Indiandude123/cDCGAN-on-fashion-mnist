import torch
import torch.nn as nn

class Discriminator(nn.Module):
  """
  Discriminator model for the GAN.
  Takes an image and classifies it as real or fake.
  The features are extracted using a CNN extractor before passing through a sigmoid activation
  """
  def __init__(self, image_size=28, num_classes=10):
    super().__init__()
    self.image_size = image_size
    self.num_classes = num_classes
    self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)
    self.linear = nn.Sequential(
      nn.Linear(self.num_classes, self.image_size*self.image_size),
      nn.LeakyReLU()
    )
    # CNN H_out = (H_in + 2*P - K)/S + 1
    self.model = nn.Sequential(
      nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1), # (2, 28, 28) to (64, 14, 14)
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (64, 14, 14) to (128, 7, 7)
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=0), # (128, 7, 7) to (256, 1, 1)
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Flatten(),
      nn.Linear(256*1*1, 1), # Flattened image into a 1d vector having 256 dimensions
      nn.Sigmoid()
    )

  def forward(self, image, labels):
    """
    Forward pass for the Discriminator.

    Args:
      image (torch.Tensor): Input image tensor.
      labels (torch.Tensor): Input class label tensor

    Returns:
      torch.Tensor: Probability that the image is real.
    """
    label_emb = self.label_embedding(labels)
    label_img = self.linear(label_emb).view(label_emb.size(0), 1, self.image_size, self.image_size)
    x = torch.cat([image, label_img], dim=1)
    x = self.model(x)
    return x

