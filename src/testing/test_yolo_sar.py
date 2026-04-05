import torch
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# PATHS
# ----------------------------

MODEL_PATH = "models-sar/yolo/best.pt"

IMAGE_DIR = "src/data/processed-d3/yolo/images/val"

LABEL_DIR = "src/data/processed-d3/yolo/labels/val"


# ----------------------------
# LOAD MODEL
# ----------------------------

print("Loading YOLO model...")

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False
)

model.conf = 0.25
model.iou = 0.45


# ----------------------------
# IOU FUNCTION
# ----------------------------

def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ----------------------------
# CONVERT YOLO LABELS
# ----------------------------

def yolo_to_xyxy(label, img_w, img_h):

    cls, xc, yc, w, h = label

    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h

    return [x1,y1,x2,y2]


# ----------------------------
# EVALUATION
# ----------------------------

TP = 0
FP = 0
FN = 0

image_files = sorted(os.listdir(IMAGE_DIR))

print("Total images:", len(image_files))

for img_name in tqdm(image_files):

    img_path = os.path.join(IMAGE_DIR, img_name)

    label_path = os.path.join(
        LABEL_DIR,
        img_name.replace(".png",".txt")
    )

    img = cv2.imread(img_path)
    h,w,_ = img.shape

    # predictions
    results = model(img_path)
    preds = results.xyxy[0].cpu().numpy()

    pred_boxes = [p[:4] for p in preds]

    # ground truth
    gt_boxes = []

    if os.path.exists(label_path):

        with open(label_path) as f:

            for line in f:

                vals = list(map(float,line.split()))

                gt_boxes.append(
                    yolo_to_xyxy(vals,w,h)
                )

    matched = set()

    for p in pred_boxes:

        found = False

        for i,g in enumerate(gt_boxes):

            if i in matched:
                continue

            if compute_iou(p,g) > 0.5:

                TP += 1
                matched.add(i)
                found = True
                break

        if not found:
            FP += 1

    FN += len(gt_boxes) - len(matched)


# ----------------------------
# METRICS
# ----------------------------

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

print("\nEvaluation Results")
print("--------------------")

print("True Positives :", TP)
print("False Positives:", FP)
print("False Negatives:", FN)

print("\nPrecision:", round(precision,4))
print("Recall   :", round(recall,4))