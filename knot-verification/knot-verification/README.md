# Knot Verification Pipeline

Decoupled, low-data pipeline for classifying a climbing knot as `Correct` /
`Incorrect` on edge hardware:

```
frame -> YOLOv10-Nano (detect + crop) -> DINOv2-Small (frozen features) -> Linear SVM (classify)
```

## Project structure

```
knot-verification/
├── config.py                        # all paths & hyperparameters live here
├── requirements.txt
├── raw_data/
│   ├── images/                      # <- put your 150 images here
│   ├── annotations.csv              # <- you create this (see schema below)
│   └── annotations_SCHEMA_EXAMPLE.csv
├── data_prep/
│   ├── make_splits.py               # stratified train/val split (shared by both stages)
│   ├── build_yolo_dataset.py        # annotations.csv -> YOLO folder format
│   └── build_classifier_dataset.py  # ground-truth crops + jitter aug -> DINOv2 features (.npz)
├── models/
│   ├── detector.py                  # KnotDetector: wraps the fine-tuned YOLOv10-N
│   ├── feature_extractor.py         # DinoV2FeatureExtractor: frozen backbone
│   └── classifier_head.py           # KnotClassifierHead: SVM or MLP, common interface
├── train_detector.py                # fine-tunes YOLOv10-N
├── train_classifier.py              # fits SVM/MLP on DINOv2 features (CV-tuned)
├── pipeline.py                      # KnotInspectionPipeline: ties all 3 stages together
├── run_on_image.py                  # CLI: test the pipeline on one image
├── evaluate.py                      # classifier CV report + end-to-end val report
└── edge_camera_loop.py              # live camera loop for the Jetson/Raspberry Pi
```

## Expected annotation format

`raw_data/annotations.csv`:

```
filename,x1,y1,x2,y2,label
figure8_0001.jpg,142,88,410,360,correct
none/empty_frame_0001.jpg,,,,,none
```

`x1,y1,x2,y2` are the knot's bounding box in absolute pixels; `label` is
`correct`/`incorrect` (or `1`/`0`). For a no-knot frame, use `label=none` and
leave all four box columns blank; it becomes an empty YOLO label file and is
used only for detector training/evaluation. If your Kaggle dataset ships a different
schema (separate label files, COCO JSON, etc.), edit
`data_prep/make_splits.py:load_annotations()` — everything downstream reads
from that one function, so it's the only place you need to adapt.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Full workflow

```bash
# 1. Split the 150 images once, reused by both stages (no leakage between them)
python data_prep/make_splits.py

# 2. Build the YOLO-format dataset and fine-tune the detector
python data_prep/build_yolo_dataset.py
python train_detector.py

# 3. Crop with ground-truth boxes + jitter augmentation, extract DINOv2 features
python data_prep/build_classifier_dataset.py

# 4. Fit the SVM head (CV-tuned regularization strength)
python train_classifier.py

# 5. Check performance
python evaluate.py

# 6. Run on a single frame, or start the live camera loop on the robot
python run_on_image.py --image raw_data/images/figure8_0001.jpg
python edge_camera_loop.py
```

## A few design choices worth knowing about

- **Classifier trains on ground-truth crops, not detector output.** This
  keeps classifier-training-signal clean and decoupled from detector
  localization error. To avoid a train/inference mismatch, `build_classifier_dataset.py`
  adds small random jitter/scale/flip crops per *training* image (validation
  crops stay clean) — this also turns 150 images into several hundred feature
  vectors at effectively zero overfitting risk, since DINOv2 stays frozen the
  whole time.
- **Evaluation uses k-fold CV, not just the small holdout.** With ~150 images,
  a single ~15% validation split is only ~20 images — too small to trust on
  its own. `train_classifier.py` and `evaluate.py` use 5-fold stratified CV
  on the training crops for the headline metric, and the held-out val split
  is still evaluated end-to-end (detector + DINOv2 + SVM together) as a
  secondary, closer-to-deployment check.
- **The pipeline returns `None` when no knot is detected** (`InspectionResult.label
  is None`) rather than forcing a classification on background — important
  so the robot doesn't confidently mislabel an empty frame.

---

## Your two questions

### Download the dataset locally, or keep it in Kaggle?

Keep it in Kaggle for the training stages, and only bring the tiny output
artifacts down to your machine:

- Reference the data at `/kaggle/input/...` in a Kaggle Notebook rather than
  downloading it — with 150 images this is more about not re-uploading than
  about size, but it also gets you Kaggle's free T4/P100 GPU for
  `train_detector.py`, which is the one step that actually benefits from a
  GPU. Feature extraction (`build_classifier_dataset.py`) and SVM training
  (`train_classifier.py`) are so cheap (150–750 forward passes through a 21M
  param frozen ViT, then an instant SVM fit) that they'd run in well under a
  minute on Kaggle's CPU too if you're out of GPU quota.
- Once training is done, you only need to move `weights/knot_detector.pt`
  (~5–10 MB) and `weights/knot_classifier.joblib` (a few KB) off Kaggle onto
  the edge device — never the raw images.
- Download a *local* copy only if you want to write/debug this code outside
  a notebook (e.g. in an IDE) before running the real training job on Kaggle.

### Is there a GitHub repo you should clone?

No — for both models here, `pip install` + auto-downloaded weights covers
it, so cloning a separate repo just adds a second, harder-to-keep-in-sync
copy of the model code:

- **YOLOv10-N**: the mainline `ultralytics` pip package natively supports
  YOLOv10 now (alongside YOLOv3/5/6/8/9, YOLO11, YOLO12, and YOLO26). `pip install ultralytics`, then `YOLO("yolov10n.pt")`
  (as used in `models/detector.py`) downloads COCO-pretrained weights
  automatically and trains normally through `model.train(...)`. You don't
  need the standalone `THU-MIG/yolov10` (or `jameslahm/yolov10`) repo.
- **DINOv2**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
  (used in `models/feature_extractor.py`) fetches the pretrained backbone
  weights on first call — no clone needed there either.

One thing worth flagging since it wasn't in your original brief: Ultralytics'
current flagship is **YOLO26**, released after YOLOv10, with end-to-end NMS-free inference and edge-deployment optimization baked in natively rather than as YOLOv10's
add-on training trick. For a from-scratch edge project today it may be worth
a quick side-by-side against YOLOv10-N — since `models/detector.py` just
takes a weights path, swapping `YOLO_BASE_WEIGHTS = "yolov10n.pt"` for
`"yolo26n.pt"` in `config.py` is a one-line experiment if you want to compare.
