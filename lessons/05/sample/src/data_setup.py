import os
import requests
import zipfile
from pathlib import Path
from utils import get_project_root
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def download_and_extract_food_dataset():
    data_path = get_project_root() / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    data_path = data_path / "pizza_steak_sushi"
    data_path.mkdir(parents=True, exist_ok=True)

    print("Saving data to:", data_path)

    with open(data_path / "food.zip", "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "food.zip", "r") as zip_ref:
        print("Extracting data...")
        zip_ref.extractall(data_path)

    os.remove(data_path / "food.zip")
    print("Download and extract complete.")

def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
    transform: transforms.Compose,
    batch_size: int = 32,
    # num_workers: int = NUM_WORKERS,
    num_workers: int = 0, # For some reason, this is causing issues on Windows with the multiprocessing module.
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create data loaders for training and testing datasets.

    Args:
        train_dir (Path): The path to the training dataset directory.
        test_dir (Path): The path to the testing dataset directory.
        transform (transforms.Compose): The transformations to apply to the images.
        batch_size (int, optional): The batch size. Defaults to 32.
        num_workers (int, optional): The number of workers to load data. Defaults to NUM_WORKERS.

    Returns:
        tuple[DataLoader, DataLoader, list[str]]: Train dataloader, test dataloader, and class names.
    """
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


if __name__ == "__main__":
    download_and_extract_food_dataset()
