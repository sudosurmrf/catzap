# Cat Identification Classifier — Design Spec

**Date:** 2026-04-08

## Overview

Add a per-cat identification layer on top of the existing YOLOv8 cat detector. Fine-tune a MobileNetV3-Small classifier to distinguish between the user's 4 cats (Bud, Gotham, Puffer 1, Renesmee) plus an "Unknown" class. Training data is collected via live-feed labeling and manual photo upload. Inference runs every ~1 second and carries identity forward between classification frames.

## Cats

| Name | Notes |
|------|-------|
| Bud | — |
| Gotham | — |
| Puffer 1 | — |
| Renesmee | AKA "the cat formally known as Stella" |

## Data Collection & Storage

### Database

Add a `cat_photos` table:

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| cat_id | UUID FK → cats | |
| file_path | TEXT | Relative path to stored image |
| source | TEXT | `'capture'` or `'upload'` |
| created_at | TIMESTAMPTZ | |

### Photo Storage

- Stored at `data/cat_photos/{cat_id}/{photo_id}.jpg`
- Crops use the detection bbox padded by 20% on each side (clamped to frame bounds)
- Original resolution and aspect ratio preserved on disk
- Resize to 224x224 via letterboxing (black bar padding) only at training/inference time

### Collection Methods

**Live capture labeling:**
- When a cat is detected on the feed, a subtle "Label" button appears near the detection
- When the classifier is uncertain (confidence 0.3–0.6), the button pulses to encourage labeling
- Clicking opens a dropdown: [Bud, Gotham, Puffer 1, Renesmee]
- Selecting a name crops the detection from the current frame (with 20% padding), saves it, links it to that cat

**Manual upload:**
- In Settings/Cats page, each cat has an "Upload Photos" button
- Accepts multiple jpg/png images
- Stores them as-is (no resize on upload)

### API Endpoints

- `POST /api/cats/{cat_id}/photos/capture` — Receives bbox coordinates + base64 frame, crops/pads/saves
- `POST /api/cats/{cat_id}/photos/upload` — Receives image files (multipart)
- `GET /api/cats/{cat_id}/photos` — List photos for a cat (returns ids, paths, source, timestamps)
- `DELETE /api/cats/{cat_id}/photos/{photo_id}` — Remove a photo (deletes file + DB record)

## Model Training Pipeline

### Architecture

- MobileNetV3-Small pre-trained on ImageNet
- Replace final classifier head with `Linear(1024, 5)`
- Outputs: [Bud, Gotham, Puffer 1, Renesmee, Unknown]

### Unknown Class

Populated by:
- Random background crops from frames where no cat was detected
- Unlabeled/skipped cat crops

### Training Flow

1. User clicks "Train Classifier" in Settings/Cats page
2. Server validates minimum 10 photos per cat
3. Freeze all layers except final classifier head + last conv block
4. Training config: Adam optimizer, lr=0.001, cross-entropy loss, 20 epochs, batch size 16
5. Data augmentation: random horizontal flip, rotation ±15°, color jitter, random crop
6. Letterbox resize to 224x224 (preserve aspect ratio, pad with black)
7. 80/20 train/validation split
8. Save best model (by validation accuracy) to `models/weights/cat_classifier.pt`
9. Update `model_version` on all cat records in database
10. Broadcast training progress over WebSocket

### Training API

- `POST /api/classifier/train` — Kicks off training as background task
- `GET /api/classifier/status` — Returns: idle | training (with progress %) | complete | error
- `GET /api/classifier/info` — Model version, accuracy, per-cat photo counts, last trained timestamp

## Inference Integration

### Classifier Module

`server/vision/classifier.py`:
- `load_model(path)` — loads trained weights, returns model
- `classify(crop: ndarray) -> (cat_name: str, confidence: float)` — letterbox resizes crop, runs inference
- Returns `("Unknown", 0.0)` if no trained model exists

### Vision Loop Integration (main.py)

- Classify every ~1 second (every 30th frame at 30fps)
- For each detection on a classification frame:
  - Crop bbox region from frame with 20% padding
  - Letterbox resize to 224x224
  - Run through classifier
  - Cache result keyed by spatial position (bbox center)
- Between classification frames:
  - Carry forward identity using nearest bbox center match to previous frame's detections
- Attach `cat_name` and `cat_confidence` to each detection dict

### Confidence Handling

- Threshold: 0.6 default (configurable in settings)
- Above 0.6: confident identification, display cat name
- Between 0.3 and 0.6: uncertain, display "Unknown Cat", show pulsing "Label?" prompt on frontend
- Below 0.3: "Unknown Cat", no prompt

### Event Logging

- `cat_name` and `cat_id` fields in events are now populated when a zap fires
- Falls back to "Unknown Cat" / null cat_id when classifier is uncertain or untrained

## Frontend Changes

### LiveFeed — Detection Labels

- Each detection overlay shows cat name (or "Unknown Cat") below the bounding box
- Uncertain detections (0.3–0.6 confidence) show a pulsing "Label?" button
- Clicking "Label?" opens dropdown anchored to that detection: [Bud, Gotham, Puffer 1, Renesmee]
- Selecting a name fires the capture endpoint, button disappears

### Settings — Cat Management

- Each cat card shows: name, photo count, sample thumbnails
- "Upload Photos" button per cat (multi-file picker, jpg/png)
- "Train Classifier" button (disabled if any cat has < 10 photos)
- Training progress bar (via WebSocket updates)
- Model info: version, accuracy, last trained timestamp

### EventLog & CatStats

- Events display cat name when available
- CatStats shows per-cat zap counts (e.g., Bud: 12, Gotham: 8)

### Type Updates

- `Detection` interface: add `cat_confidence?: number` (already has optional `cat_name`)
- `FrameData`: no changes needed (detections array already carries through)

## Component Structure

### New Server Files

- `server/vision/classifier.py` — Model loading, inference, letterbox preprocessing
- `server/vision/trainer.py` — Training pipeline (data loading, augmentation, training loop, model save)
- `server/routers/classifier.py` — Train/status/info endpoints
- `server/routers/photos.py` — Photo CRUD (capture, upload, list, delete)

### Modified Server Files

- `server/models/database.py` — Add cat_photos table, photo CRUD functions
- `server/main.py` — Integrate classifier into vision loop
- `server/config.py` — Already has `cat_photos_dir` and `classifier_weights_dir`

### New Frontend Files

- `frontend/src/components/CatLabelDropdown.tsx` — Dropdown for labeling detections on the feed

### Modified Frontend Files

- `frontend/src/components/LiveFeed.tsx` — Show cat names on detections, render label prompt
- `frontend/src/components/Settings.tsx` — Cat management with photo upload, training controls
- `frontend/src/types/index.ts` — Add `cat_confidence` to Detection
- `frontend/src/api/client.ts` — Add photo and classifier API functions
