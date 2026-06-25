from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WoundDataset(Dataset):
    """Dataset PyTorch basé sur les CSV de split."""

    def __init__(self, dataframe, transform=None, class_to_idx=None, project_root=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.project_root = Path(
            project_root) if project_root is not None else Path.cwd()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = Path(row["path"])
        class_name = row["class"]

        # Rend les chemins robustes depuis la racine du projet.
        if not image_path.is_absolute():
            if str(image_path).startswith("../data/"):
                image_path = self.project_root / \
                    str(image_path).replace("../data/", "data/")
            else:
                image_path = self.project_root / image_path

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[class_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224):
    """Construit les transformations train et validation/test."""
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        # Augmentations modérées adaptées à l'imagerie de plaies.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transforms, eval_transforms


def load_splits(split_dir):
    """Charge les fichiers CSV train, validation et test."""
    split_dir = Path(split_dir)

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    return train_df, val_df, test_df


def build_class_mapping(train_df):
    """Crée les mappings classe <-> indice à partir du train."""
    class_names = sorted(train_df["class"].unique())
    class_to_idx = {class_name: idx for idx,
                    class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name,
                    idx in class_to_idx.items()}

    return class_names, class_to_idx, idx_to_class


def compute_class_weights(train_df, class_names):
    """Calcule un poids plus élevé pour les classes minoritaires."""
    class_counts = train_df["class"].value_counts().sort_index()
    class_counts = class_counts[class_names]

    num_classes = len(class_names)
    num_train_samples = len(train_df)

    class_weights = num_train_samples / (num_classes * class_counts)

    return class_weights


def build_weighted_sampler(train_df, class_weights):
    """Crée un sampler pour rééquilibrer les batches d'entraînement."""
    sample_weights = train_df["class"].map(class_weights.to_dict()).values

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights.copy(), dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def build_dataloaders(
    split_dir,
    batch_size=16,
    img_size=224,
    use_weighted_sampler=False,
    num_workers=0,
    project_root=None
):
    """Prépare les DataLoaders et les métadonnées de classification."""
    train_df, val_df, test_df = load_splits(split_dir)

    class_names, class_to_idx, idx_to_class = build_class_mapping(train_df)

    train_transforms, eval_transforms = get_transforms(img_size=img_size)

    train_dataset = WoundDataset(
        dataframe=train_df,
        transform=train_transforms,
        class_to_idx=class_to_idx,
        project_root=project_root
    )

    val_dataset = WoundDataset(
        dataframe=val_df,
        transform=eval_transforms,
        class_to_idx=class_to_idx,
        project_root=project_root
    )

    test_dataset = WoundDataset(
        dataframe=test_df,
        transform=eval_transforms,
        class_to_idx=class_to_idx,
        project_root=project_root
    )

    if use_weighted_sampler:
        class_weights = compute_class_weights(train_df, class_names)
        sampler = build_weighted_sampler(train_df, class_weights)

        # Avec un sampler, on ne met pas shuffle=True.
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        class_weights = compute_class_weights(train_df, class_names)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    metadata = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_weights": class_weights
    }

    return train_loader, val_loader, test_loader, metadata
