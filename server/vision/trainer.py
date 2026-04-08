import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from server.vision.classifier import letterbox

logger = logging.getLogger(__name__)


class CatPhotoDataset(Dataset):
    """Loads cat photos from disk with augmentation."""

    def __init__(self, samples: list[tuple[str, int]], augment: bool = True):
        self.samples = samples
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        image = cv2.imread(file_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        letterboxed = letterbox(rgb, 224)

        from PIL import Image
        pil_image = Image.fromarray(letterboxed)

        if self.augment:
            tensor = self.aug_transform(pil_image)
        else:
            tensor = self.transform(pil_image)

        return tensor, label


class TrainingStatus:
    """Shared training status for polling."""
    def __init__(self):
        self.state: str = "idle"
        self.progress: float = 0.0
        self.accuracy: float = 0.0
        self.error: str | None = None


training_status = TrainingStatus()


def train_classifier(
    cat_photo_map: dict[str, list[str]],
    unknown_photos: list[str],
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
) -> bool:
    global training_status
    training_status.state = "training"
    training_status.progress = 0.0
    training_status.error = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class_names = sorted(cat_photo_map.keys()) + ["Unknown"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        num_classes = len(class_names)

        all_samples: list[tuple[str, int]] = []
        for cat_name, paths in cat_photo_map.items():
            idx = class_to_idx[cat_name]
            all_samples.extend((p, idx) for p in paths)
        unknown_idx = class_to_idx["Unknown"]
        all_samples.extend((p, unknown_idx) for p in unknown_photos)

        if len(all_samples) < 10:
            training_status.state = "error"
            training_status.error = "Not enough training data"
            return False

        random.shuffle(all_samples)
        split = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split]
        val_samples = all_samples[split:]

        train_dataset = CatPhotoDataset(train_samples, augment=True)
        val_dataset = CatPhotoDataset(val_samples, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True

        model.to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    _, predicted = torch.max(output, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_acc = correct / max(total, 1)
            training_status.progress = ((epoch + 1) / epochs) * 100
            training_status.accuracy = val_acc
            logger.info(f"Epoch {epoch + 1}/{epochs} — val_acc: {val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "accuracy": val_acc,
                }, output_path)

        training_status.state = "complete"
        training_status.accuracy = best_val_acc
        logger.info(f"Training complete — best val_acc: {best_val_acc:.3f}")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status.state = "error"
        training_status.error = str(e)
        return False
