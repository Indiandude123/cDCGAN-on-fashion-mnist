import argparse
import os
import sys

# Add the project root to the Python path to allow importing modules
# This assumes main.py is run from the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the main functions from your training and generation scripts
from train import run_training
from generate import run_generation

def main():
    """
    Main entry point for the cDCGAN project.
    Uses argparse to handle different commands (train, generate).
    """
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST conditional DCGAN Training and Generation CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train command ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train the GAN model.",
        description="""
        Train the cDCGAN on the Fashion-MNIST dataset.
        Example:
            python main.py train --epochs 100 
            python main.py train --epochs 50 --num_steps_gen 3 --num_steps_disc 1
        """
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100"
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training. Default: 128"
    )
    train_parser.add_argument(
        "--num_steps_gen",
        type=int,
        default="1",
        help="Number of steps of the generator to train. Default: 1"
    )
    train_parser.add_argument(
        "--num_steps_disc",
        type=int,
        default="1",
        help="Number of steps of the discriminator to train. Default: 1"
    )


    # --- Generate command ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate samples from a trained cDCGAN.",
        description="""
        Generate and display images using a trained Generator model.
        Example:
            python main.py generate
            python main.py generate --model_path checkpoints/generator_epoch_100.pth
        """
    )
    generate_parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("checkpoints", "generator_final.pth"),
        help="Path to the trained generator model (.pth file). "
             "Default: checkpoints/generator_final.pth"
    )
    generate_parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate and display. Default: 16"
    )

    args = parser.parse_args()

    if args.command == "train":
        print(f"Starting cDCGAN training for {args.epochs} epochs ...")
        run_training(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_steps_gen=args.num_steps_gen,
            num_steps_disc=args.num_steps_dic,
        )
    elif args.command == "generate":
        print(f"Generating {args.num_samples} samples using model: {args.model_path}...")
        run_generation(
            model_path=args.model_path,
            num_samples=args.num_samples
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

