import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torchvision.utils as vutils # For saving image grids

def view_samples(title_prefix, samples):
    """
    Displays a grid of generated samples using Matplotlib.

    Args:
        title_prefix (str): A string to prefix the plot title (e.g., "Final Generated Samples").
        samples (torch.Tensor): A batch of generated images.
                                Expected shape: (N, C, H, W) e.g., (16, 1, 28, 28).
                                Images are expected to be in [0, 1] range for proper display.
    """
    # Create a figure and a grid of subplots (4x4 for 16 samples)
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)

    # Convert the tensor to a NumPy array and move to CPU
    # Ensure it's detached from the computational graph
    samples_np = samples.detach().cpu().numpy()

    # Iterate through the flattened axes and the samples
    for ax, img in zip(axes.flatten(), samples_np):
        ax.xaxis.set_visible(False) # Hide x-axis ticks
        ax.yaxis.set_visible(False) # Hide y-axis ticks

        # Handle different image channel dimensions (grayscale vs. RGB)
        if img.shape[0] == 1: # Grayscale image (C, H, W) -> (H, W)
            # Reshape to (Height, Width) for imshow if it's (1, H, W)
            im = ax.imshow(img.reshape((img.shape[1], img.shape[2])), cmap='Greys_r')
        elif img.shape[0] == 3: # RGB image (C, H, W) -> (H, W, C)
            # Transpose to (Height, Width, Channels) for matplotlib
            im = ax.imshow(np.transpose(img, (1, 2, 0)))
        else:
            raise ValueError(f"Unsupported image channel dimension: {img.shape[0]}. Expected 1 or 3.")

    # Add a title to the entire figure
    fig.suptitle(f'{title_prefix} Generated Samples', fontsize=16)
    # Adjust layout to make space for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Display the plot

def save_generated_images(epoch, generator, fixed_noise, fixed_labels, results_dir="results"):
    """
    Generates images using the generator and saves them to a directory.

    Args:
        epoch (int): The current epoch number (for naming the saved file).
        generator (torch.nn.Module): The trained Generator model.
        fixed_noise (torch.Tensor): A fixed noise vector to generate consistent samples.
        fixed_labels (torch.Tensor): A vector of fixed class labels corresponding to samples
        results_dir (str): Directory to save the generated images.
    """
    # Set generator to evaluation mode (important for consistent behavior, especially with BatchNorm/Dropout)
    generator.eval()

    # Create the output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad(): # No need to compute gradients for visualization
        # Generate images from the fixed noise
        fake_images = generator(fixed_noise, fixed_labels).cpu() # Move to CPU for saving

        # Rescale from [-1, 1] to [0, 1] for proper display/saving since generator output is Tanh
        fake_images = fake_images * 0.5 + 0.5

    # Save the image grid directly using torchvision.utils.save_image
    # nrow specifies the number of images in each row of the grid
    save_path = os.path.join(results_dir, f"epoch_{epoch:03d}.png")
    vutils.save_image(fake_images, save_path, nrow=8) # Changed nrow to 8 for a more square grid (16 samples)

    # Set generator back to training mode if applicable (e.g., if called within a training loop)
    generator.train()

