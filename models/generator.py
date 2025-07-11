import torch
import torch.nn as nn

class Generator(nn.Module):
  """
  Generator model for the GAN.
  Takes a noise vector, a random class label and generates an image corresponding to that class label.
  """
  def __init__(self, noise_dim, image_size=28, num_classes=10):
    super().__init__()
    self.image_size = image_size
    self.label_embedding = nn.Embedding(num_classes, num_classes) #num_embeddings = num_classes and embedding_dim = num_classes
    input_dim = noise_dim + num_classes
    # For transposed convolutions the following formula holds
    # output = (input-1)*stride - 2*input_padding + kernel + output_padding
    self.linear = nn.Sequential(
      nn.Linear(input_dim, 1*1*256),
      nn.ReLU()
    )
    self.model = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=7, stride=1, padding=0), # (256,1,1) to (128,7,7)
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=0), #(128,7,7) to (64,11,11)
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0), #(64,11,11) to (32,14,14)
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=0), #(32,14,14) to (16,17,17)
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.ConvTranspose2d(16, 8, kernel_size=4, stride=1, padding=0), #(16,17,17) to (8,20,20),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.ConvTranspose2d(8, 4, kernel_size=4, stride=1, padding=0), #(8,20,20) to (4,23,23),
      nn.BatchNorm2d(4),
      nn.ReLU(),
      nn.ConvTranspose2d(4, 2, kernel_size=3, stride=1, padding=0), #(4,23,23) to (2,25,25),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      nn.ConvTranspose2d(2, 1, kernel_size=4, stride=1, padding=0), #(2,25,25) to (1,28,28),
      nn.Tanh()
    )

  def forward(self, noise, labels):
    x = torch.cat([noise, self.label_embedding(labels)], dim=1)
    x = self.linear(x)
    x = x.view(x.size(0), 256, 1, 1)
    x = self.model(x)
    return x
