import torch
import os

# Import models and visualization utilities
from models.generator import Generator
from utils.visualize import view_samples

# --- Configuration (defaults, can be overridden by main.py) ---
NOISE_DIM = 100
IMAGE_DIM = 28  # For Fashion-MNIST (1 channel, 28x28 pixels)
NUM_CLASSES = 10
CHECKPOINTS_DIR = "checkpoints"

def run_generation(model_path, num_samples):
    """
    Loads a trained generator, generates new samples, and displays them.

    Args:
        model_path (str): Path to the trained generator model (.pth file).
        num_samples (int): Number of samples to generate and display.
    """
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate your Generator model
    generator = Generator(NOISE_DIM, IMAGE_DIM, NUM_CLASSES).to(device)

    # 2. Load the trained generator model's state dictionary
    if not os.path.exists(model_path):
        print(f"Error: Trained generator model not found at {model_path}.")
        print("Please ensure you have trained the GAN using 'python main.py train' and the model is saved.")
        return

    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Generator model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading generator model: {e}")
        return

    # Set the generator to evaluation mode
    generator.eval()

    # 3. Generate a new set of random noise vectors
    z = torch.randn(num_samples, NOISE_DIM).to(device)
    
    # 4. Generate a set of random class labels for which images have to be generated
    gen_labels = torch.randint(0, NUM_CLASSES, (num_samples,)).to(device)

    # 4. Pass these noise vectors and class labels through the generator
    with torch.no_grad(): # No need to compute gradients for visualization
        sampled_generated_images = generator(z, gen_labels).cpu() # Generate and move to CPU

    # 5. Rescale images from [-1, 1] to [0, 1] if your generator's last activation is Tanh
    # This is crucial for correct display with matplotlib
    sampled_generated_images = sampled_generated_images * 0.5 + 0.5

    # 6. Call the view_samples function to display
    view_samples("Generated Samples", sampled_generated_images)

if __name__ == "__main__":
    # If generate.py is run directly, use default parameters
    run_generation(
        model_path=os.path.join(CHECKPOINTS_DIR, 'generator_final.pth'),
        num_samples=16
    )
