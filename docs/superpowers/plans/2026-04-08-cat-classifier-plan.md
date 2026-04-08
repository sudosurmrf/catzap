# Cat Identification Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a MobileNetV3-Small classifier to identify which of the user's 4 cats (Bud, Gotham, Puffer 1, Renesmee) was detected, with training data collected via live feed labeling and manual upload.

**Architecture:** Two-phase pipeline — YOLOv8 detects cats (existing), then MobileNetV3 classifies which cat. Training data collected via photo capture from the live feed or manual upload. Training runs locally as a background task. Inference runs every ~1 second and carries identity forward between frames.

**Tech Stack:** Python 3.11+, PyTorch (torchvision MobileNetV3), FastAPI, asyncpg, React/TypeScript

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `server/vision/classifier.py` | Model loading, letterbox preprocessing, inference |
| Create | `server/vision/trainer.py` | Training pipeline (data loading, augmentation, train loop, save) |
| Create | `server/routers/photos.py` | Photo capture/upload/list/delete endpoints |
| Create | `server/routers/classifier.py` | Train/status/info endpoints |
| Create | `frontend/src/components/CatLabelDropdown.tsx` | Dropdown for labeling detections on feed |
| Modify | `server/models/database.py:11-63` | Add cat_photos table + CRUD functions |
| Modify | `server/main.py:1-26,170-270` | Integrate classifier into vision loop |
| Modify | `server/main.py:60-80` | Register new routers |
| Modify | `server/config.py:14` | Add classifier_confidence_threshold setting |
| Modify | `frontend/src/types/index.ts:37-41` | Add cat_confidence to Detection |
| Modify | `frontend/src/api/client.ts:39-49` | Add photo + classifier API functions |
| Modify | `frontend/src/components/Settings.tsx:69-93` | Add photo count, upload, train button per cat |
| Modify | `frontend/src/components/LiveFeed.tsx` | Show cat names + label prompt on detections |

---

### Task 1: Database — cat_photos Table and CRUD

**Files:**
- Modify: `server/models/database.py:11-63` (SCHEMA string)
- Modify: `server/models/database.py` (add functions at end)

- [ ] **Step 1: Add cat_photos table to SCHEMA**

In `server/models/database.py`, add after the `cats` table definition (after line 33, before the `events` table):

```python
CREATE TABLE IF NOT EXISTS cat_photos (
    id UUID PRIMARY KEY,
    cat_id UUID NOT NULL REFERENCES cats(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 2: Add photo CRUD functions**

Add at the end of `server/models/database.py`:

```python
async def create_cat_photo(
    cat_id: str,
    file_path: str,
    source: str = "upload",
    conn: asyncpg.Connection | None = None,
) -> str:
    photo_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO cat_photos (id, cat_id, file_path, source)
               VALUES ($1, $2, $3, $4)""",
            photo_id, uuid.UUID(cat_id), file_path, source,
        )
        return str(photo_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_cat_photos(cat_id: str, conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch(
            "SELECT * FROM cat_photos WHERE cat_id = $1 ORDER BY created_at DESC",
            uuid.UUID(cat_id),
        )
        return [
            {
                "id": str(row["id"]),
                "cat_id": str(row["cat_id"]),
                "file_path": row["file_path"],
                "source": row["source"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def delete_cat_photo(photo_id: str, conn: asyncpg.Connection | None = None) -> str | None:
    """Delete a photo record and return its file_path for cleanup, or None if not found."""
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        row = await c.fetchrow(
            "DELETE FROM cat_photos WHERE id = $1 RETURNING file_path",
            uuid.UUID(photo_id),
        )
        return row["file_path"] if row else None
    finally:
        if conn is None:
            await pool.release(c)


async def get_cat_photo_counts(conn: asyncpg.Connection | None = None) -> dict[str, int]:
    """Return {cat_id: photo_count} for all cats."""
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch(
            "SELECT cat_id, COUNT(*) as count FROM cat_photos GROUP BY cat_id"
        )
        return {str(row["cat_id"]): row["count"] for row in rows}
    finally:
        if conn is None:
            await pool.release(c)
```

- [ ] **Step 3: Verify schema applies**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -c "import server.models.database; print('import ok')"`
Expected: `import ok`

---

### Task 2: Photo API Router

**Files:**
- Create: `server/routers/photos.py`

- [ ] **Step 1: Create photos router**

Create `server/routers/photos.py`:

```python
import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from server.config import settings
from server.models.database import (
    create_cat_photo, get_cat_photos, delete_cat_photo, get_cats,
)

router = APIRouter(prefix="/api/cats", tags=["photos"])


class CaptureRequest(BaseModel):
    frame_base64: str
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1


def _crop_with_padding(frame: np.ndarray, bbox: list[float], pad_pct: float = 0.2) -> np.ndarray:
    """Crop bbox region from frame with padding, preserving aspect ratio."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    # Add 20% padding on each side
    x1 = max(0, x1 - bw * pad_pct)
    y1 = max(0, y1 - bh * pad_pct)
    x2 = min(1, x2 + bw * pad_pct)
    y2 = min(1, y2 + bh * pad_pct)
    # Convert to pixel coords
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return frame[py1:py2, px1:px2]


def _save_photo(cat_id: str, photo_id: str, image: np.ndarray) -> str:
    """Save image to disk and return relative path."""
    photo_dir = settings.cat_photos_dir / cat_id
    photo_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{cat_id}/{photo_id}.jpg"
    abs_path = settings.cat_photos_dir / rel_path
    cv2.imwrite(str(abs_path), image)
    return rel_path


@router.post("/{cat_id}/photos/capture", status_code=201)
async def capture_photo(cat_id: str, req: CaptureRequest):
    """Capture a cat photo from the live feed by cropping a detection bbox."""
    cats = await get_cats()
    if not any(c["id"] == cat_id for c in cats):
        raise HTTPException(status_code=404, detail="Cat not found")

    # Decode base64 frame
    frame_bytes = base64.b64decode(req.frame_base64)
    frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame data")

    crop = _crop_with_padding(frame, req.bbox)
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="Empty crop region")

    photo_id = str(uuid.uuid4())
    rel_path = _save_photo(cat_id, photo_id, crop)
    db_id = await create_cat_photo(cat_id=cat_id, file_path=rel_path, source="capture")
    return {"id": db_id, "file_path": rel_path}


@router.post("/{cat_id}/photos/upload", status_code=201)
async def upload_photos(cat_id: str, files: list[UploadFile] = File(...)):
    """Upload one or more cat photos."""
    cats = await get_cats()
    if not any(c["id"] == cat_id for c in cats):
        raise HTTPException(status_code=404, detail="Cat not found")

    results = []
    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            continue  # skip invalid images

        photo_id = str(uuid.uuid4())
        rel_path = _save_photo(cat_id, photo_id, image)
        db_id = await create_cat_photo(cat_id=cat_id, file_path=rel_path, source="upload")
        results.append({"id": db_id, "file_path": rel_path})

    return results


@router.get("/{cat_id}/photos")
async def list_photos(cat_id: str):
    return await get_cat_photos(cat_id)


@router.delete("/{cat_id}/photos/{photo_id}")
async def remove_photo(cat_id: str, photo_id: str):
    file_path = await delete_cat_photo(photo_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    # Delete file from disk
    abs_path = settings.cat_photos_dir / file_path
    if abs_path.exists():
        abs_path.unlink()
    return {"deleted": True}
```

- [ ] **Step 2: Register router in main.py**

In `server/main.py`, add to the imports (around line 14):

```python
from server.routers import zones, cats, events, control, stream, settings as settings_router, spatial
```

Add after that line:

```python
from server.routers import photos as photos_router, classifier as classifier_router
```

Then in the `app` setup section where routers are included (search for `app.include_router`), add:

```python
app.include_router(photos_router.router)
app.include_router(classifier_router.router)
```

Note: The classifier router doesn't exist yet — it will be created in Task 5. If this causes an import error, comment out the classifier_router lines until Task 5.

---

### Task 3: Classifier Module — Inference

**Files:**
- Create: `server/vision/classifier.py`

- [ ] **Step 1: Create classifier module**

Create `server/vision/classifier.py`:

```python
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Letterbox resize: pad to square then resize to target, preserving aspect ratio
def letterbox(image: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CatClassifier:
    def __init__(self):
        self.model: nn.Module | None = None
        self.class_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, weights_path: Path) -> bool:
        """Load a trained classifier. Returns True if successful."""
        if not weights_path.exists():
            logger.info("No classifier weights found at %s", weights_path)
            return False
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.class_names = checkpoint["class_names"]
            num_classes = len(self.class_names)

            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Loaded classifier with classes: %s", self.class_names)
            return True
        except Exception as e:
            logger.error("Failed to load classifier: %s", e)
            self.model = None
            return False

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Classify a cat crop. Returns (cat_name, confidence).
        Returns ("Unknown", 0.0) if no model is loaded."""
        if self.model is None:
            return ("Unknown", 0.0)

        # BGR to RGB, letterbox, transform
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        letterboxed = letterbox(rgb, 224)
        tensor = _transform(letterboxed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = torch.max(probs, dim=1)
            cat_name = self.class_names[idx.item()]
            return (cat_name, float(confidence.item()))
```

- [ ] **Step 2: Verify import**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -c "from server.vision.classifier import CatClassifier; print('ok')"`
Expected: `ok`

---

### Task 4: Training Pipeline

**Files:**
- Create: `server/vision/trainer.py`

- [ ] **Step 1: Create the trainer module**

Create `server/vision/trainer.py`:

```python
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
        """samples: list of (file_path, class_index)"""
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
            # Return a black image if file is missing
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
        self.state: str = "idle"  # idle | training | complete | error
        self.progress: float = 0.0  # 0-100
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
    """Train the MobileNetV3 classifier.

    Args:
        cat_photo_map: {cat_name: [file_path, ...]} for each known cat.
        unknown_photos: List of file paths for the "Unknown" class.
        output_path: Where to save the trained model weights.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.

    Returns:
        True if training succeeded.
    """
    global training_status
    training_status.state = "training"
    training_status.progress = 0.0
    training_status.error = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build class list: known cats + Unknown
        class_names = sorted(cat_photo_map.keys()) + ["Unknown"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        num_classes = len(class_names)

        # Build samples list
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

        # 80/20 split
        random.shuffle(all_samples)
        split = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split]
        val_samples = all_samples[split:]

        train_dataset = CatPhotoDataset(train_samples, augment=True)
        val_dataset = CatPhotoDataset(val_samples, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model: MobileNetV3-Small with custom head
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last conv block (features[-1]) + classifier
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
            # Train
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            # Validate
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
```

- [ ] **Step 2: Verify import**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -c "from server.vision.trainer import train_classifier, training_status; print('ok')"`
Expected: `ok`

---

### Task 5: Classifier API Router

**Files:**
- Create: `server/routers/classifier.py`

- [ ] **Step 1: Create classifier router**

Create `server/routers/classifier.py`:

```python
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from server.config import settings
from server.models.database import get_cats, get_cat_photos, get_cat_photo_counts
from server.vision.trainer import train_classifier, training_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classifier", tags=["classifier"])

WEIGHTS_PATH = settings.classifier_weights_dir / "cat_classifier.pt"
MIN_PHOTOS_PER_CAT = 10


@router.post("/train")
async def start_training():
    """Kick off classifier training as a background task."""
    if training_status.state == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    cats = await get_cats()
    if not cats:
        raise HTTPException(status_code=400, detail="No cats registered")

    # Validate minimum photos per cat
    photo_counts = await get_cat_photo_counts()
    for cat in cats:
        count = photo_counts.get(cat["id"], 0)
        if count < MIN_PHOTOS_PER_CAT:
            raise HTTPException(
                status_code=400,
                detail=f"{cat['name']} has {count} photos (need {MIN_PHOTOS_PER_CAT})",
            )

    # Build cat_photo_map: {cat_name: [abs_file_paths]}
    cat_photo_map: dict[str, list[str]] = {}
    for cat in cats:
        photos = await get_cat_photos(cat["id"])
        cat_photo_map[cat["name"]] = [
            str(settings.cat_photos_dir / p["file_path"]) for p in photos
        ]

    # For now, unknown_photos is empty — can be extended later
    unknown_photos: list[str] = []

    # Run training in a thread (blocking PyTorch ops)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        train_classifier,
        cat_photo_map,
        unknown_photos,
        WEIGHTS_PATH,
    )

    return {"status": "training_started"}


@router.get("/status")
async def get_training_status():
    return {
        "state": training_status.state,
        "progress": training_status.progress,
        "accuracy": training_status.accuracy,
        "error": training_status.error,
    }


@router.get("/info")
async def get_classifier_info():
    photo_counts = await get_cat_photo_counts()
    cats = await get_cats()

    per_cat = []
    for cat in cats:
        per_cat.append({
            "name": cat["name"],
            "photo_count": photo_counts.get(cat["id"], 0),
        })

    return {
        "model_exists": WEIGHTS_PATH.exists(),
        "weights_path": str(WEIGHTS_PATH),
        "per_cat": per_cat,
        "min_photos_required": MIN_PHOTOS_PER_CAT,
    }
```

- [ ] **Step 2: Verify import**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -c "from server.routers.classifier import router; print('ok')"`
Expected: `ok`

---

### Task 6: Config Update + Classifier Confidence Setting

**Files:**
- Modify: `server/config.py:14`

- [ ] **Step 1: Add classifier settings**

In `server/config.py`, after line 14 (`confidence_threshold: float = 0.5`), add:

```python
    # Classifier
    classifier_confidence_threshold: float = 0.6
    classifier_uncertain_min: float = 0.3
    classify_every_n_frames: int = 30
```

---

### Task 7: Vision Loop Integration

**Files:**
- Modify: `server/main.py:1-26` (imports)
- Modify: `server/main.py:170-270` (vision loop)

- [ ] **Step 1: Add classifier to imports and shared state**

In `server/main.py`, add to imports (after line 16):

```python
from server.vision.classifier import CatClassifier
```

Add to shared state (after line 35, after `_cat_tracker`):

```python
_classifier: CatClassifier | None = None
_classify_frame_counter: int = 0
_cached_identities: dict[str, tuple[str, float]] = {}  # det_key -> (name, conf)
```

- [ ] **Step 2: Initialize classifier on startup**

In the startup/lifespan code where other components are initialized, add:

```python
global _classifier
_classifier = CatClassifier()
weights_path = settings.classifier_weights_dir / "cat_classifier.pt"
_classifier.load(weights_path)
```

- [ ] **Step 3: Add classification to the detection loop**

In the vision loop, after the `detections = detector.detect(frame)` line (line 178) and before zone checking, add:

```python
            # Classify cats periodically
            global _classify_frame_counter, _cached_identities
            _classify_frame_counter += 1
            should_classify = (_classify_frame_counter % settings.classify_every_n_frames == 0)

            if should_classify and _classifier is not None and _classifier.model is not None:
                new_identities: dict[str, tuple[str, float]] = {}
                for i, det in enumerate(detections):
                    bbox = det["bbox"]
                    # Crop with 20% padding
                    h, w = frame.shape[:2]
                    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x1 = max(0, int((bbox[0] - bw * 0.2) * w))
                    y1 = max(0, int((bbox[1] - bh * 0.2) * h))
                    x2 = min(w, int((bbox[2] + bw * 0.2) * w))
                    y2 = min(h, int((bbox[3] + bh * 0.2) * h))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        name, conf = _classifier.classify(crop)
                        if conf >= settings.classifier_confidence_threshold:
                            det["cat_name"] = name
                            det["cat_confidence"] = conf
                        elif conf >= settings.classifier_uncertain_min:
                            det["cat_name"] = "Unknown"
                            det["cat_confidence"] = conf
                        else:
                            det["cat_name"] = "Unknown"
                            det["cat_confidence"] = 0.0
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        key = f"{round(cx, 1)}_{round(cy, 1)}"
                        new_identities[key] = (det.get("cat_name", "Unknown"), det.get("cat_confidence", 0.0))
                _cached_identities = new_identities
            else:
                # Carry forward cached identities by nearest match
                for det in detections:
                    bbox = det["bbox"]
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    best_key = None
                    best_dist = float("inf")
                    for key, (name, conf) in _cached_identities.items():
                        kx, ky = (float(v) for v in key.split("_"))
                        dist = (cx - kx) ** 2 + (cy - ky) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_key = key
                    if best_key is not None and best_dist < 0.05:
                        name, conf = _cached_identities[best_key]
                        det["cat_name"] = name
                        det["cat_confidence"] = conf
```

- [ ] **Step 4: Add classifier reload function**

Add a function accessible from the classifier router so after training completes, the model is reloaded:

```python
def reload_classifier():
    """Reload classifier weights after training."""
    global _classifier
    if _classifier is not None:
        weights_path = settings.classifier_weights_dir / "cat_classifier.pt"
        _classifier.load(weights_path)
```

In `server/routers/classifier.py`, after training completes in the `start_training` endpoint, the `run_in_executor` callback should trigger a reload. Update the training dispatch:

```python
    async def _train_and_reload():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            train_classifier,
            cat_photo_map,
            unknown_photos,
            WEIGHTS_PATH,
        )
        # Reload the model in main
        from server.main import reload_classifier
        reload_classifier()

    asyncio.create_task(_train_and_reload())

    return {"status": "training_started"}
```

---

### Task 8: Frontend Types + API Client Updates

**Files:**
- Modify: `frontend/src/types/index.ts:37-41`
- Modify: `frontend/src/api/client.ts:39-49`

- [ ] **Step 1: Update Detection type**

In `frontend/src/types/index.ts`, update the Detection interface (line 37-41):

```typescript
export interface Detection {
  bbox: number[];
  confidence: number;
  cat_name?: string;
  cat_confidence?: number;
}
```

- [ ] **Step 2: Add photo + classifier API functions**

In `frontend/src/api/client.ts`, add after the existing cats section (after line 49):

```typescript
// Cat Photos
export const getCatPhotos = (catId: string) =>
  fetchJSON<{ id: string; file_path: string; source: string; created_at: string }[]>(
    `/cats/${catId}/photos`
  );

export const capturePhoto = (catId: string, frameBase64: string, bbox: number[]) =>
  fetch(`${API_BASE}/cats/${catId}/photos/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame_base64: frameBase64, bbox }),
  }).then((r) => r.json());

export const uploadPhotos = (catId: string, files: FileList) => {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  return fetch(`${API_BASE}/cats/${catId}/photos/upload`, {
    method: "POST",
    body: form,
  }).then((r) => r.json());
};

export const deleteCatPhoto = (catId: string, photoId: string) =>
  fetchJSON(`/cats/${catId}/photos/${photoId}`, { method: "DELETE" });

// Classifier
export const startTraining = () =>
  fetchJSON<{ status: string }>("/classifier/train", { method: "POST" });

export const getTrainingStatus = () =>
  fetchJSON<{ state: string; progress: number; accuracy: number; error: string | null }>(
    "/classifier/status"
  );

export const getClassifierInfo = () =>
  fetchJSON<{
    model_exists: boolean;
    per_cat: { name: string; photo_count: number }[];
    min_photos_required: number;
  }>("/classifier/info");
```

---

### Task 9: CatLabelDropdown Component

**Files:**
- Create: `frontend/src/components/CatLabelDropdown.tsx`

- [ ] **Step 1: Create the dropdown component**

Create `frontend/src/components/CatLabelDropdown.tsx`:

```tsx
import { useState, useEffect } from "react";
import type { Cat } from "../types";
import { getCats, capturePhoto } from "../api/client";

interface CatLabelDropdownProps {
  bbox: number[];
  frameBase64: string;
  onLabeled: () => void;
  style?: React.CSSProperties;
}

export default function CatLabelDropdown({ bbox, frameBase64, onLabeled, style }: CatLabelDropdownProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [open, setOpen] = useState(false);
  const [labeling, setLabeling] = useState(false);

  useEffect(() => {
    getCats().then(setCats).catch(console.error);
  }, []);

  async function handleSelect(catId: string) {
    setLabeling(true);
    try {
      await capturePhoto(catId, frameBase64, bbox);
      onLabeled();
    } catch (e) {
      console.error("Failed to label:", e);
    }
    setLabeling(false);
    setOpen(false);
  }

  if (!open) {
    return (
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
        style={{
          padding: "2px 6px",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          background: "rgba(245, 158, 11, 0.2)",
          border: "1px solid var(--amber-dim)",
          borderRadius: 3,
          color: "var(--amber)",
          cursor: "pointer",
          animation: "pulse 2s infinite",
          ...style,
        }}
      >
        Label?
      </button>
    );
  }

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        background: "var(--bg-panel)",
        border: "1px solid var(--border-base)",
        borderRadius: "var(--radius-sm)",
        padding: 4,
        minWidth: 100,
        ...style,
      }}
    >
      {cats.map((cat) => (
        <button
          key={cat.id}
          onClick={() => handleSelect(cat.id)}
          disabled={labeling}
          style={{
            padding: "3px 8px",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            background: "var(--bg-deep)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 3,
            color: "var(--text-secondary)",
            cursor: labeling ? "wait" : "pointer",
            textAlign: "left",
          }}
        >
          {cat.name}
        </button>
      ))}
      <button
        onClick={() => setOpen(false)}
        style={{
          padding: "2px 6px",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          background: "transparent",
          border: "none",
          color: "var(--text-ghost)",
          cursor: "pointer",
          textAlign: "center",
        }}
      >
        cancel
      </button>
    </div>
  );
}
```

---

### Task 10: Settings Page — Photo Upload + Training Controls

**Files:**
- Modify: `frontend/src/components/Settings.tsx`

- [ ] **Step 1: Update imports and add state**

Replace the imports at the top of `Settings.tsx`:

```typescript
import { useEffect, useState, useRef } from "react";
import type { Cat } from "../types";
import {
  getCats, createCat, deleteCat,
  uploadPhotos, getClassifierInfo, startTraining, getTrainingStatus,
} from "../api/client";
import CalibrationWizard from "./CalibrationWizard";
```

Add state inside the component, after the existing `newCatName` state:

```typescript
  const [photoCounts, setPhotoCounts] = useState<Record<string, number>>({});
  const [trainingState, setTrainingState] = useState<string>("idle");
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingAccuracy, setTrainingAccuracy] = useState(0);
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});
```

- [ ] **Step 2: Add data loading and training poll**

Add after the existing `useEffect` that loads cats:

```typescript
  useEffect(() => {
    getClassifierInfo().then((info) => {
      const counts: Record<string, number> = {};
      info.per_cat.forEach((c) => { counts[c.name] = c.photo_count; });
      setPhotoCounts(counts);
    }).catch(console.error);
  }, [cats]);

  useEffect(() => {
    if (trainingState !== "training") return;
    const interval = setInterval(async () => {
      const status = await getTrainingStatus();
      setTrainingState(status.state);
      setTrainingProgress(status.progress);
      setTrainingAccuracy(status.accuracy);
    }, 1000);
    return () => clearInterval(interval);
  }, [trainingState]);

  async function handleUpload(catId: string, files: FileList | null) {
    if (!files || files.length === 0) return;
    await uploadPhotos(catId, files);
    const info = await getClassifierInfo();
    const counts: Record<string, number> = {};
    info.per_cat.forEach((c) => { counts[c.name] = c.photo_count; });
    setPhotoCounts(counts);
  }

  async function handleTrain() {
    setTrainingState("training");
    setTrainingProgress(0);
    try {
      await startTraining();
    } catch (e: any) {
      setTrainingState("error");
    }
  }

  const allCatsReady = cats.length > 0 && cats.every((c) => (photoCounts[c.name] || 0) >= 10);
```

- [ ] **Step 3: Update cat card rendering**

Replace the cat card map block (lines 69-93 of the original Settings.tsx) with:

```tsx
        {cats.map((cat) => (
          <div
            key={cat.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "7px 10px",
              background: "var(--bg-deep)",
              borderRadius: "var(--radius-sm)",
              marginBottom: 3,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "var(--text-secondary)" }}>{cat.name}</span>
              <span style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                {photoCounts[cat.name] || 0} photos
              </span>
            </div>
            <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
              <input
                type="file"
                accept="image/*"
                multiple
                ref={(el) => { fileInputRefs.current[cat.id] = el; }}
                style={{ display: "none" }}
                onChange={(e) => handleUpload(cat.id, e.target.files)}
              />
              <button
                className="btn btn-sm"
                onClick={() => fileInputRefs.current[cat.id]?.click()}
                style={{ padding: "2px 8px", fontSize: 10 }}
              >
                Upload
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={() => handleDeleteCat(cat.id)}
                style={{ padding: "2px 8px", fontSize: 10 }}
              >
                Remove
              </button>
            </div>
          </div>
        ))}

        {/* Train button */}
        {cats.length > 0 && (
          <div style={{ marginTop: 8 }}>
            {trainingState === "training" ? (
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <div style={{ color: "var(--amber)", marginBottom: 4 }}>
                  Training... {trainingProgress.toFixed(0)}%
                </div>
                <div style={{
                  height: 4, background: "var(--bg-elevated)", borderRadius: 2, overflow: "hidden",
                }}>
                  <div style={{
                    width: `${trainingProgress}%`, height: "100%",
                    background: "var(--amber)", borderRadius: 2,
                    transition: "width 0.3s ease",
                  }} />
                </div>
              </div>
            ) : (
              <button
                className="btn btn-primary btn-sm"
                onClick={handleTrain}
                disabled={!allCatsReady || trainingState === "training"}
                style={{ width: "100%", opacity: allCatsReady ? 1 : 0.4 }}
              >
                {trainingState === "complete"
                  ? `Retrain (${(trainingAccuracy * 100).toFixed(0)}% acc)`
                  : "Train Classifier"}
              </button>
            )}
            {!allCatsReady && cats.length > 0 && (
              <div style={{ fontSize: 9, color: "var(--text-ghost)", marginTop: 4, textAlign: "center" }}>
                Need at least 10 photos per cat to train
              </div>
            )}
          </div>
        )}
```

---

### Task 11: LiveFeed — Cat Name Labels + Label Prompt

**Files:**
- Modify: `frontend/src/components/LiveFeed.tsx`

- [ ] **Step 1: Add CatLabelDropdown import**

At the top of `LiveFeed.tsx`, add:

```typescript
import CatLabelDropdown from "./CatLabelDropdown";
```

- [ ] **Step 2: Add frame capture ref**

Add a ref to store the latest base64 frame for photo capture. Near the existing refs:

```typescript
const latestFrameRef = useRef<string>("");
```

In the WebSocket `onmessage` handler, after the `img.src = ...` line, add:

```typescript
latestFrameRef.current = data.frame;
```

- [ ] **Step 3: Add cat name labels to detection overlay**

In the `drawFrame` function where detections are drawn on the canvas, the detection bounding boxes are rendered. After drawing each bbox, add the cat name text. Find the section that draws detection boxes and add:

```typescript
// Draw cat name label
const catName = det.cat_name || "Unknown";
const catConf = det.cat_confidence ?? 0;
ctx.fillStyle = catConf >= 0.6 ? "rgba(16, 185, 129, 0.9)" : "rgba(161, 161, 170, 0.7)";
ctx.font = "bold 11px 'IBM Plex Mono'";
ctx.fillText(catName, x1 * w + 4, y2 * h - 4);
```

- [ ] **Step 4: Add label prompt overlay for uncertain detections**

In the JSX return, add an overlay for uncertain detections. After the existing detection-related overlays, add:

```tsx
      {/* Cat label prompts for uncertain detections */}
      {!drawMode && detections.map((det, i) => {
        const conf = det.cat_confidence ?? 0;
        if (conf < 0.3 || conf >= 0.6) return null;  // only show for uncertain range
        const [x1, y1, x2, y2] = det.bbox;
        const cx = ((x1 + x2) / 2) * 100;
        const cy = (y2) * 100;
        return (
          <CatLabelDropdown
            key={`label-${i}`}
            bbox={det.bbox}
            frameBase64={latestFrameRef.current}
            onLabeled={() => {}}
            style={{
              position: "absolute",
              left: `${cx}%`,
              top: `${cy}%`,
              transform: "translate(-50%, 4px)",
              zIndex: 15,
            }}
          />
        );
      })}
```

---

### Task 12: Manual Testing Checklist

- [ ] **Step 1: Start the server and verify endpoints**

Start the server. Verify:
- `GET /api/classifier/info` returns per-cat photo counts
- `GET /api/classifier/status` returns `{"state": "idle", ...}`

- [ ] **Step 2: Test photo upload**

In the Settings page, add a cat, then click "Upload" and select images. Verify the photo count increments.

- [ ] **Step 3: Test live capture labeling**

On the LiveFeed, when a cat is detected with uncertain confidence, click "Label?" and select a cat name. Verify the photo is saved.

- [ ] **Step 4: Test training**

Upload 10+ photos for each cat. Click "Train Classifier". Verify the progress bar advances and training completes.

- [ ] **Step 5: Test inference**

After training, verify that detected cats show names on the LiveFeed bounding boxes instead of "Unknown".

---
