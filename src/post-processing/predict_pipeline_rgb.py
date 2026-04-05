# /content/drive/MyDrive/OIL-SPILL-8/src/post-processing/predict_pipeline.py

# D1 SPECIFIC INFERENCE PIPELINE

import os
import cv2
import numpy as np
import torch
import tensorflow as tf

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = "."

YOLO_WEIGHTS = f"{BASE_DIR}/models-d1/yolo/best.pt"
CNN_MODEL_PATH = f"{BASE_DIR}/models-d1/cnn/cnn_classifier.keras"
UNET_MODEL_PATH = f"{BASE_DIR}/models-d1/unet/unet_segmentation.keras"

TEST_DIR = f"{BASE_DIR}/src/data/processed/test/images"
RESULTS_DIR = f"{BASE_DIR}/results/inference"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

# Thresholds
CNN_THRESH_YOLO_PASS = 0.70
CNN_THRESH_YOLO_FAIL = 0.92
UNET_MIN_AREA_RATIO = 0.005
UNET_MAX_AREA_RATIO = 0.60

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

print("📌 Loading models...")

yolo_model = torch.hub.load(
    f"{BASE_DIR}/yolov5",
    "custom",
    path=YOLO_WEIGHTS,
    source="local",
    force_reload=False
)

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

unet_model = tf.keras.models.load_model(
    UNET_MODEL_PATH,
    compile=False
)

print("✅ All models loaded successfully")

# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------

def preprocess_for_cnn_unet(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm

# ---------------------------------------------------
# UNET VALIDATION
# ---------------------------------------------------

def validate_and_clean_mask(mask_prob):

    mask_bin = (mask_prob > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    clean = np.zeros_like(mask_bin)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(clean, [cnt], -1, 1, -1)

    area_ratio = np.sum(clean) / (clean.shape[0] * clean.shape[1])
    print(f"📏 UNet area ratio: {area_ratio:.4f}")

    if area_ratio < UNET_MIN_AREA_RATIO:
        print("❌ Rejected: Too small region")
        return None

    if area_ratio > UNET_MAX_AREA_RATIO:
        print("❌ Rejected: Too large region")
        return None

    return clean

# ---------------------------------------------------
# CREATE BBOX IMAGE
# ---------------------------------------------------

def create_bbox_image(img_proc, img_bgr, detections, clean_mask, run_unet):

    bbox_img = img_proc.copy()

    if run_unet:

        if len(detections) > 0:
            scale_x = IMG_SIZE[0] / img_bgr.shape[1]
            scale_y = IMG_SIZE[1] / img_bgr.shape[0]

            for det in detections:
                x1, y1, x2, y2, conf, cls = det

                rx1 = int(x1 * scale_x)
                ry1 = int(y1 * scale_y)
                rx2 = int(x2 * scale_x)
                ry2 = int(y2 * scale_y)

                cv2.rectangle(bbox_img, (rx1, ry1), (rx2, ry2), (0, 1, 0), 2)

        elif clean_mask is not None:
            ys, xs = np.where(clean_mask == 1)
            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (1, 1, 0), 2)

    return bbox_img

# ---------------------------------------------------
# SAVE 2x2 GRID
# ---------------------------------------------------

def save_square_output(original, bbox_img, heatmap, overlay, filename):

    original_u8 = (original * 255).astype(np.uint8)
    bbox_u8 = (bbox_img * 255).astype(np.uint8)

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay_u8 = (overlay * 255).astype(np.uint8)

    top_row = np.concatenate([original_u8, bbox_u8], axis=1)
    bottom_row = np.concatenate([heatmap_color, overlay_u8], axis=1)

    square = np.concatenate([top_row, bottom_row], axis=0)

    save_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(square, cv2.COLOR_RGB2BGR))

    print(f"💾 Saved → {save_path}")

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------

def predict_all():

    yolo_raw_success = 0
    fused_success = 0
    total_images = 0

    image_list = sorted(os.listdir(TEST_DIR))

    for fname in image_list:

        if not fname.lower().endswith((".jpg", ".png")):
            continue

        total_images += 1
        print("\n=================================================")
        print(f"🖼 Processing: {fname}")

        img_path = os.path.join(TEST_DIR, fname)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print("❌ Could not read image")
            continue

        img_proc = preprocess_for_cnn_unet(img_bgr)

        # ---------------- YOLO ----------------
        yolo_results = yolo_model(img_bgr)
        detections = yolo_results.xyxy[0].cpu().numpy()
        yolo_detected = len(detections) > 0
        print(f"🔍 YOLO detected: {yolo_detected}")

        if yolo_detected:
            yolo_raw_success += 1

        # ---------------- CNN (Full Image for fallback) ----------------
        cnn_conf = cnn_model.predict(img_proc[None, ...], verbose=0)[0][0]
        print(f"🧠 CNN full-image confidence: {cnn_conf:.3f}")

        # ---------------- REGION-BASED UNET ----------------
        full_mask = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.uint8)
        run_unet = False

        if yolo_detected:

            scale_x = IMG_SIZE[0] / img_bgr.shape[1]
            scale_y = IMG_SIZE[1] / img_bgr.shape[0]

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_proc = preprocess_for_cnn_unet(crop)

                crop_conf = cnn_model.predict(
                    crop_proc[None, ...], verbose=0
                )[0][0]

                print(f"🧠 CNN (box) confidence: {crop_conf:.3f}")

                if crop_conf > CNN_THRESH_YOLO_PASS:

                    run_unet = True

                    unet_prob_crop = unet_model.predict(
                        crop_proc[None, ...], verbose=0
                    )[0, ..., 0]

                    clean_crop = validate_and_clean_mask(unet_prob_crop)

                    if clean_crop is not None:

                        mask_resized = cv2.resize(
                            clean_crop.astype(np.uint8),
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_NEAREST
                        )

                        rx1 = int(x1 * scale_x)
                        ry1 = int(y1 * scale_y)
                        rx2 = int(x2 * scale_x)
                        ry2 = int(y2 * scale_y)

                        mask_resized = cv2.resize(
                            mask_resized,
                            (rx2 - rx1, ry2 - ry1),
                            interpolation=cv2.INTER_NEAREST
                        )

                        full_mask[ry1:ry2, rx1:rx2] = mask_resized

        else:
            if cnn_conf > CNN_THRESH_YOLO_FAIL:

                run_unet = True

                unet_prob_full = unet_model.predict(
                    img_proc[None, ...], verbose=0
                )[0, ..., 0]

                clean_full = validate_and_clean_mask(unet_prob_full)

                if clean_full is not None:
                    full_mask = clean_full

        # ---------------- Overlay ----------------
        overlay = img_proc.copy()
        if np.sum(full_mask) > 0:
            overlay[full_mask == 1] = [1, 0, 0]
            fused_success += 1

        clean_mask = full_mask if np.sum(full_mask) > 0 else None
        unet_prob = full_mask.astype(np.float32)

        # ---------------- BBOX IMAGE ----------------
        bbox_img = create_bbox_image(
            img_proc, img_bgr, detections, clean_mask, run_unet
        )

        # ---------------- SAVE ----------------
        save_square_output(
            img_proc,
            bbox_img,
            unet_prob,
            overlay,
            fname
        )

    print("\n=======================================")
    print(f"📊 YOLO Raw Detection Rate: {yolo_raw_success}/{total_images}")
    print(f"📊 Final Fused Positive Rate: {fused_success}/{total_images}")
    print("=======================================")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    predict_all()
    print("\n✅ Batch inference completed successfully")
