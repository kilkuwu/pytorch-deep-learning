"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import argparse

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001


def main():
    parser = argparse.ArgumentParser(
        prog="Trainer",
        usage="%(prog)s [options]",
        description="Train a PyTorch image classification model.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of epochs to train the model (default: 200)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training and testing (default: 64)",
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=HIDDEN_UNITS,
        help="Number of hidden units in the model (default: 10)",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for the optimizer (default: 0.0001)",
    )

    args = parser.parse_args()

    # Setup directories
    train_dir = utils.get_project_root() / "data/pizza_steak_sushi/train"
    test_dir = utils.get_project_root() / "data/pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )

    # Create DataLoaders with help from data_setup.py
    print("Creating data loaders...")
    train_dataloader, test_dataloader, class_names = (
        data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=data_transform,
            batch_size=args.batch_size,
        )
    )
    print(f"Created dataloaders. Number of classes: {len(class_names)}")

    # Create model with help from model_builder.py
    print("Creating model...")
    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=args.hidden_units, output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    print("Setting loss function and optimizer...")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training with help from engine.py
    print("Starting training...")
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device,
    )

    # Save the model with help from utils.py
    utils.save_model(
        model=model,
        target_dir=utils.get_project_root() / "models",
        model_name="05_tinyvgg_model.pth",
    )


if __name__ == "__main__":
    main()
