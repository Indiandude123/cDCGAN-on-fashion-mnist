import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.utils as vutils # For saving generated images

# Import models and utilities
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_fashion_mnist_dataloader
from utils.visualize import save_generated_images # Removed view_samples as it's for generate.py

# --- Configuration (defaults, can be overridden by main.py) ---
DATA_PATH = "data/fashion-mnist_train.csv"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_training(num_epochs, batch_size, num_steps_gen=1, num_steps_disc=1):
  """
  Trains the GAN (Generator and Discriminator). Mention the number of steps of generator and discriminator
  to train in a batch. By default the number of steps for training generator and discriminator is set to 1.

  Args:
    num_epochs (int): Number of epochs to train.
    batch_size (int): Batch size for the DataLoader.
    num_steps_gen (int): Number of steps of generator to train. 
    num_steps_disc (int): Number of steps of discriminator to train.
    
  """
  # --- Model Parameters ---
  NOISE_DIM = 100
  IMAGE_DIM = 28 # For Fashion-MNIST (1 channel, 28x28 pixels)
  NUM_CLASSES = 10
  
  # --- Model Instantiation ---
  generator = Generator(NOISE_DIM, IMAGE_DIM, NUM_CLASSES).to(device)
  discriminator = Discriminator(IMAGE_DIM, NUM_CLASSES).to(device)

  # --- Optimizers ---
  gen_optimizer = optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
  disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.000015, betas=(0.5, 0.999))

  # --- Loss Function ---
  criterion = nn.BCELoss() # Binary Cross-Entropy Loss for GANs

  # --- Create directories ---
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
  os.makedirs(RESULTS_DIR, exist_ok=True)

  # Get DataLoader
  train_loader = get_fashion_mnist_dataloader(DATA_PATH, batch_size)

  # Fixed noise for visualizing generator's progress over epochs
  fixed_noise = torch.randn(64, NOISE_DIM).to(device) # 64 samples for visualization
  fixed_labels = torch.randint(0, NUM_CLASSES, (64,)).to(device) # 64 class labels for which sample images to be generated

  history_disc_losses = []
  history_gen_losses = []

  print(f'Starting GAN training for {num_epochs}')

  for epoch in range(num_epochs):
    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    num_batches = 0
    for batch_idx, (real_images, real_labels) in enumerate(train_loader):
      batch_size = real_images.size(0)
      real_images = real_images.to(device)
      real_labels = real_labels.to(device)

      real = torch.ones(batch_size, 1).to(device)
      fake = torch.zeros(batch_size, 1).to(device)

      ####### Train Discriminator ####### 
      for _ in range(num_steps_disc):
        # 1st loss term of discriminator loss due to real images
        disc_output = discriminator(real_images, real_labels)
        disc_real_loss = criterion(disc_output, real)
        
        # 2nd loss term of discriminator loss due to fake images
        z = torch.randn(batch_size, NOISE_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
        with torch.no_grad():
            fake_gen_images = generator(z, fake_labels)
            
        disc_output = discriminator(fake_gen_images.detach(), fake_labels)
        disc_fake_loss = criterion(disc_output, fake)

        disc_loss = disc_real_loss + disc_fake_loss
        
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

      ####### Train Generator ####### 
      for _ in range(num_steps_gen):
        # 1st loss term of the overall loss term doesn't depend upon theta params of genertor,
        # hence we only optimize over the 2nd loss term
        z = torch.randn(batch_size, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
        gen_images = generator(z, gen_labels)
        disc_output = discriminator(gen_images, gen_labels)

        gen_loss = criterion(disc_output, real)

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

      epoch_disc_loss += disc_loss.item()
      epoch_gen_loss += gen_loss.item()
      num_batches += 1

      # Optional: Print progress within epoch
      if (batch_idx + 1) % 100 == 0: # Print every 100 batches
          print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                f'D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}')

    # Average losses for the epoch
    avg_disc_loss = epoch_disc_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    history_disc_losses.append(avg_disc_loss)
    history_gen_losses.append(avg_gen_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Avg D Loss: {avg_disc_loss:.4f}, Avg G Loss: {avg_gen_loss:.4f}')
    
    # Save generated images periodically
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs: # Save at intervals and at the end
        print(f'--- Saving generated images for Epoch {epoch+1} ---')
        save_generated_images(epoch+1, generator, fixed_noise, fixed_labels, results_dir=RESULTS_DIR)
        
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'generator_epoch_{epoch+1:03d}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'discriminator_epoch_{epoch+1:03d}.pth'))
        print(f"Models saved at epoch {epoch+1}")

  print("Training finished.")
  # Save final models
  torch.save(generator.state_dict(), os.path.join(CHECKPOINTS_DIR, 'generator_final.pth'))
  torch.save(discriminator.state_dict(), os.path.join(CHECKPOINTS_DIR, 'discriminator_final.pth'))
  print("Final models saved.")

  return history_disc_losses, history_gen_losses

if __name__ == "__main__":
    # If train.py is run directly, use default parameters
    run_training(
        num_epochs=100,
        batch_size=128,
        num_steps_gen=1, 
        num_steps_disc=1
    )
