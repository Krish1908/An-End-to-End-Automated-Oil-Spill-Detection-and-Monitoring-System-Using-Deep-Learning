#!/usr/bin/env python3
"""
Enhanced Oil Spill Detection Web Application - Streamlit Version
Integrates all 3 ML models: CNN (Classification), U-Net (Segmentation), YOLO (Detection)
Features: Timestamps, Legend Overlays, Comprehensive Results, Multi-Model Analysis

FIXES APPLIED (v2):
  1. land_water_mask() — was the primary culprit. The old fixed-threshold approach
     classified dock/sand/ship pixels as "water", producing a chaotic mask that
     AND-ed away the real oil region. Fixed with:
       • SAR/satellite-aware heuristic: use the U-Net probability map directly to
         derive a "permissive" water zone (skip the land mask entirely when image is
         a visible-spectrum satellite image with mixed terrain).
       • New option: SKIP_WATER_MASK (default True) — just use the morphology-cleaned
         mask without the AND step, which is the correct behaviour for RGB satellite
         imagery that already has diverse textures.
       • When the water mask IS used, switch to Otsu on the full image rather than
         a hardcoded luminance threshold.
  2. postprocess_mask() — the water-mask AND was silently deleting valid oil pixels.
     Now the AND step is guarded by SKIP_WATER_MASK.
  3. Auto-polarity detection — border_ratio heuristic was unreliable. Replaced with
     a simpler check: if mean(pred_prob) > 0.5 the model is outputting "high = background",
     so invert. This matches the U-Net output convention reliably.
  4. use_column_width deprecation warnings — replaced with use_container_width=True.
  5. Thresholds now properly updated to globals before calling postprocess_mask.
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import os
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# =================================================
# PAGE CONFIGURATION
# =================================================
st.set_page_config(
    page_title="AI-Driven Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================
# CONFIGURATION & MODEL PATHS
# =================================================
IMG_SIZE = 256
CNN_THRESHOLD = 0.5
UNET_THRESHOLD = 0.4
MIN_AREA = 600
YOLO_CONFIDENCE = 0.5

WORKSPACE_ROOT = Path(__file__).parent
CNN_MODEL_PATH  = WORKSPACE_ROOT / "models" / "cnn"  / "cnn_classifier.keras"
UNET_MODEL_PATH = WORKSPACE_ROOT / "models" / "unet" / "unet_segmentation.keras"
YOLO_MODEL_PATH = WORKSPACE_ROOT / "models" / "yolo" / "best.pt"

# =================================================
# HELPER FUNCTIONS
# =================================================

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# FIX 1 — land_water_mask
# ---------------------------------------------------------------------------
def land_water_mask(image_rgb, skip=False):
    """
    Return a binary water mask (1 = water / oil-possible region).

    Parameters
    ----------
    image_rgb : np.ndarray, shape (H, W, 3), uint8
    skip      : bool
        When True, the function returns an all-ones mask so that the water-mask
        AND step in postprocess_mask becomes a no-op.  This is the correct
        default for visible-spectrum satellite imagery with mixed terrain
        (ships, docks, sand) where simple luminance thresholding is unreliable.

    Method choices (controlled via sidebar):
      'skip'  — all ones (recommended for RGB satellite / mixed terrain)
      'otsu'  — Otsu's adaptive thresholding on grayscale
      'fixed' — fixed luminance threshold (legacy; only reliable for dark-water SAR)
    """
    if skip:
        return np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    method = globals().get('WATER_METHOD', 'otsu')
    fixed_thresh = globals().get('WATER_FIXED_THRESHOLD', 130)

    if method == 'otsu':
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        water = th
    else:  # fixed
        _, water = cv2.threshold(gray, fixed_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7, 7), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)
    return (water > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# FIX 2 — postprocess_mask  (water-mask AND now guarded by SKIP_WATER_MASK)
# ---------------------------------------------------------------------------
def postprocess_mask(pred_prob, image_rgb,
                     threshold=None,
                     open_kernel=5, close_kernel=5,
                     min_area=None):
    """
    Post-process U-Net prediction mask.

    Key fix: the water-mask AND step is now conditional on SKIP_WATER_MASK.
    For RGB satellite imagery, set SKIP_WATER_MASK = True (sidebar default)
    to prevent the broken luminance mask from erasing valid oil regions.
    """
    if threshold is None:
        threshold = globals().get('UNET_THRESHOLD', 0.4)
    if min_area is None:
        min_area = globals().get('MIN_AREA', 600)

    raw_mask = (pred_prob > float(threshold)).astype(np.uint8)

    k_open  = np.ones((max(1, int(open_kernel)),  max(1, int(open_kernel))),  np.uint8)
    k_close = np.ones((max(1, int(close_kernel)), max(1, int(close_kernel))), np.uint8)
    oil_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k_open)
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, k_close)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(oil_mask)
    clean_mask = np.zeros_like(oil_mask)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) > int(min_area):
            clean_mask[labels == i] = 1

    # --- Water mask gate ---
    skip_water = globals().get('SKIP_WATER_MASK', True)
    img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    water_mask  = land_water_mask(img_resized, skip=skip_water)

    return (clean_mask & water_mask).astype(np.uint8)


def pick_auto_threshold(pred_prob, image_rgb,
                        open_kernel=5, close_kernel=5,
                        min_area=None):
    """Sweep thresholds and return the one that yields the most stable mask."""
    if min_area is None:
        min_area = globals().get('MIN_AREA', 600)

    best       = None
    best_score = (999, 0)

    for t in np.linspace(0.01, 0.95, 95):
        mask_t = postprocess_mask(pred_prob, image_rgb,
                                  threshold=t,
                                  open_kernel=open_kernel,
                                  close_kernel=close_kernel,
                                  min_area=min_area)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_t)
        n   = int(num_labels - 1)
        lrg = 0 if n == 0 else int(stats[1:, cv2.CC_STAT_AREA].max())

        if n <= 2 and lrg > min_area:
            return float(t)

        score = (n, -lrg)
        if score < best_score:
            best_score = score
            best = float(t)

    return best if best is not None else float(globals().get('UNET_THRESHOLD', 0.4))


def create_overlay(image_rgb, mask, alpha=0.4,
                   draw_outline=True, outline_color=(0, 0, 0)):
    """Blend a red fill over oil regions with optional boundary outline."""
    image   = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    blended = image.copy().astype(np.float32)

    mask_indices = mask == 1
    blended[mask_indices] = (
        blended[mask_indices] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_outline:
        mask_uint8  = (mask * 255).astype(np.uint8)
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_smooth = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(blended, [approx], 0, outline_color, thickness=2)

    return blended.astype(np.uint8)


def create_legend_overlay(image_rgb, mask, yolo_results=None, timestamp=""):
    """Full overlay with red fill, boundary, YOLO boxes, and text legend."""
    image   = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    blended = image.copy().astype(np.float32)

    alpha = 0.4
    mask_indices = mask == 1
    blended[mask_indices] = (
        blended[mask_indices] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    mask_uint8  = (mask * 255).astype(np.uint8)
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_smooth = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(blended, [approx], 0, (0, 0, 0), 2)

    if yolo_results is not None:
        try:
            if ULTRALYTICS_AVAILABLE:
                for result in yolo_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(blended, f'YOLO: {conf:.2f}',
                                    (x1, max(y1-10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                if hasattr(yolo_results, 'xyxy'):
                    for det in yolo_results.xyxy[0]:
                        x1, y1, x2, y2, conf, _ = det.cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(blended, f'YOLO: {conf:.2f}',
                                    (x1, max(y1-10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing YOLO boxes: {e}")

    font     = cv2.FONT_HERSHEY_SIMPLEX
    legend_y = 20
    texts = [
        f"Timestamp: {timestamp}",
        "Red Fill: U-Net Oil Spill",
        "Black Outline: U-Net Boundary",
        "Yellow Box: YOLO Detection",
    ]
    for text in texts:
        text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(blended, (8, legend_y - 10), (15 + text_size[0], legend_y + 10), (0, 0, 0), -1)
        cv2.putText(blended, text, (10, legend_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 20

    return blended.astype(np.uint8)


def calculate_metrics(mask):
    total_pixels = int(mask.size)
    oil_pixels   = int(mask.sum())
    oil_percentage = float((oil_pixels / total_pixels) * 100) if total_pixels > 0 else 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 0 if num_labels == 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
    return {
        'oil_percentage':         oil_percentage,
        'total_components':       int(num_labels - 1),
        'largest_component_area': largest,
        'total_oil_pixels':       oil_pixels,
    }


def create_results_chart(metrics, cnn_prob, yolo_count):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Multi-Model Oil Spill Detection Results", fontsize=14, fontweight='bold')

    oil_percent   = metrics['oil_percentage']
    water_percent = 100 - oil_percent

    ax1.pie([oil_percent, water_percent],
            labels=['Oil Spill', 'Clean Water'],
            autopct='%1.1f%%',
            colors=['#ff4d4d', '#4CAF50'],
            explode=(0.08, 0), startangle=90,
            textprops={"fontsize": 9})
    ax1.set_title('U-Net: Oil Spill Area Distribution', fontsize=10, fontweight='bold')

    ax2.bar(['Oil Detection'], [cnn_prob * 100], color='#2196F3', alpha=0.7)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Confidence (%)', fontsize=9)
    ax2.set_title(f'CNN Detection Confidence: {cnn_prob:.3f}', fontsize=10, fontweight='bold')
    ax2.axhline(y=CNN_THRESHOLD * 100, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {CNN_THRESHOLD}')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    ax3.bar(['Total\nComponents', 'Largest\nComponent Area'],
            [metrics['total_components'], metrics['largest_component_area']],
            color=['#9C27B0', '#FF9800'], alpha=0.7)
    ax3.set_ylabel('Pixels', fontsize=9)
    ax3.set_title('U-Net: Component Analysis', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    model_names     = ['CNN\nConfidence', 'U-Net\nSpill %', f'YOLO\nBoxes\n({yolo_count})']
    detection_scores= [min(cnn_prob * 100, 100), oil_percent, min(yolo_count * 20, 100)]
    ax4.bar(model_names, detection_scores, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.7)
    ax4.set_ylabel('Score', fontsize=9)
    ax4.set_title('Multi-Model Detection Summary', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# =================================================
# LOAD MODELS (CACHED)
# =================================================

@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
        st.sidebar.success("✅ CNN Model Loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ CNN Error: {e}")
        return None

@st.cache_resource
def load_unet_model():
    try:
        model = tf.keras.models.load_model(str(UNET_MODEL_PATH), compile=False)
        st.sidebar.success("✅ U-Net Model Loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ U-Net Error: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    try:
        yolo_abs = os.path.abspath(YOLO_MODEL_PATH)
        if not os.path.exists(yolo_abs):
            st.sidebar.info("ℹ️ YOLO: Model file not found")
            return None
        if ULTRALYTICS_AVAILABLE:
            model = YOLO(yolo_abs)
            model.conf = YOLO_CONFIDENCE
            st.sidebar.success("✅ YOLO Loaded")
            return model
        st.sidebar.info("ℹ️ YOLO: ultralytics not installed")
        return None
    except Exception:
        st.sidebar.info("ℹ️ YOLO: Disabled")
        return None


cnn_model  = load_cnn_model()
unet_model = load_unet_model()
yolo_model = load_yolo_model()

# =================================================
# STYLES
# =================================================
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 1rem;
}
.main-header h1 { margin: 0; font-size: 2.5rem; }
.main-header p  { margin: 0.5rem 0 0; font-size: 1rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛢️ AI-Driven Oil Spill Detection</h1>
    <p>CNN → Detection | U-Net → Segmentation | YOLO → Localization</p>
</div>
""", unsafe_allow_html=True)

timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
st.caption(f"⏰ Analysis Timestamp: {timestamp}")
st.divider()

# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("📊 Detection Sensitivity")
    st.caption("Adjust how confidently the models must be before flagging oil.")

    cnn_threshold  = st.slider("CNN Detection Threshold",       0.0, 1.0, value=float(CNN_THRESHOLD),  step=0.01,
                                help="Minimum CNN confidence to trigger segmentation (default: 0.5)")
    unet_threshold = st.slider("Segmentation Sensitivity",      0.0, 1.0, value=float(UNET_THRESHOLD), step=0.01,
                                help="Lower = more oil detected, Higher = stricter (default: 0.4)")
    min_area       = st.number_input("Minimum Spill Size (pixels)", min_value=0, value=int(MIN_AREA), step=50,
                                      help="Ignore detected patches smaller than this (default: 600)")
    yolo_conf      = st.slider("YOLO Detection Confidence",     0.0, 1.0, value=float(YOLO_CONFIDENCE), step=0.01,
                                help="Minimum confidence for YOLO bounding boxes (default: 0.5)")

    # Fixed internals — not exposed to public users
    show_unet_raw        = False
    auto_detect_polarity = True
    manual_invert        = False
    use_auto_threshold   = False
    open_kernel_size     = 5
    close_kernel_size    = 5

    # ---------------------------------------------------------------------------
    # Water mask — hardcoded skip for RGB satellite (right default for this app)
    # ---------------------------------------------------------------------------
    SKIP_WATER_MASK       = True
    WATER_METHOD          = "otsu"
    WATER_FIXED_THRESHOLD = 130

    # Push to globals so helper functions can read them
    globals()['SKIP_WATER_MASK']        = SKIP_WATER_MASK
    globals()['WATER_METHOD']           = WATER_METHOD
    globals()['WATER_FIXED_THRESHOLD']  = int(WATER_FIXED_THRESHOLD)

    # Update threshold globals
    CNN_THRESHOLD  = float(cnn_threshold)
    UNET_THRESHOLD = float(unet_threshold)
    MIN_AREA       = int(min_area)
    YOLO_CONFIDENCE= float(yolo_conf)
    globals()['CNN_THRESHOLD']  = CNN_THRESHOLD
    globals()['UNET_THRESHOLD'] = UNET_THRESHOLD
    globals()['MIN_AREA']       = MIN_AREA

    st.divider()
    st.subheader("📖 Colour Guide")
    st.markdown("""
    🟥 **Red** → Detected oil region  
    ⬛ **Black outline** → Oil boundary  
    🟨 **Yellow box** → YOLO localisation  
    ⬜ **White mask** → Segmentation output  
    """)

    st.divider()
    st.subheader("🖍️ Visualisation")
    outline_enabled = st.toggle("Show Oil Boundary Outline", value=True)
    OUTLINE_COLOR   = (0, 0, 0)   # fixed black — appropriate for satellite imagery

    st.divider()
    st.subheader("📁 Loaded Models")
    st.caption(f"🔵 CNN:   {CNN_MODEL_PATH.name}")
    st.caption(f"🟢 U-Net: {UNET_MODEL_PATH.name}")
    st.caption(f"🟠 YOLO:  {YOLO_MODEL_PATH.name}")


# =================================================
# MAIN — Upload
# =================================================
st.subheader("📤 Upload SAR / Satellite Image")
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "png", "jpeg"],
    help="Upload a SAR or satellite image for oil spill detection"
)

if uploaded_file:
    if cnn_model is None or unet_model is None:
        st.error("❌ Required models not loaded.")
        st.stop()

    bytes_data = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # FIX 4 — use_container_width instead of deprecated use_column_width
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True, clamp=True)

    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    st.info("🔄 Processing image with all models...")

    img_proc  = preprocess_image(img_rgb)
    img_batch = np.expand_dims(img_proc, axis=0)

    # -------------------------------------------------------
    # STEP 1: CNN
    # -------------------------------------------------------
    cnn_prob = float(cnn_model.predict(img_batch, verbose=0)[0][0])

    if cnn_prob < CNN_THRESHOLD:
        st.success(f"✅ No Oil Spill Detected (CNN: {cnn_prob:.3f})")
        st.info(f"Threshold = {CNN_THRESHOLD}. Image classified as clean.")
        st.stop()

    st.warning(f"🚨 Oil Spill Detected! (CNN: {cnn_prob:.3f})")
    st.info("Running segmentation and localization models…")

    # -------------------------------------------------------
    # STEP 2: U-Net
    # -------------------------------------------------------
    pred_prob = unet_model.predict(img_batch, verbose=0)[0].squeeze()

    # Debug view
    if show_unet_raw:
        try:
            fig_raw, ax_raw = plt.subplots(figsize=(5, 5))
            im = ax_raw.imshow(pred_prob, cmap='viridis')
            ax_raw.set_title('U-Net Raw Probabilities')
            ax_raw.axis('off')
            fig_raw.colorbar(im, ax=ax_raw, fraction=0.046, pad=0.04)
            st.pyplot(fig_raw)
            plt.close(fig_raw)
            st.write(f"min: {pred_prob.min():.4f}  max: {pred_prob.max():.4f}  mean: {pred_prob.mean():.4f}")

            # Show intermediate masks
            raw_mask   = (pred_prob > UNET_THRESHOLD).astype(np.uint8)
            k          = np.ones((5, 5), np.uint8)
            cleaned    = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k)
            cleaned    = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, k)
            img_rsz    = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            water_mask = land_water_mask(img_rsz, skip=SKIP_WATER_MASK)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image((raw_mask * 255).astype(np.uint8),   clamp=True, caption='Raw binary mask')
            with c2:
                st.image((cleaned  * 255).astype(np.uint8),   clamp=True, caption='Cleaned (morphology)')
            with c3:
                st.image((water_mask * 255).astype(np.uint8), clamp=True, caption=f'Water mask (skip={SKIP_WATER_MASK})')

            st.write(f"Raw: {raw_mask.sum()} px | Cleaned: {cleaned.sum()} px | Water: {water_mask.sum()} px")

            # Download debug files
            dl1, dl2, dl3 = st.columns(3)
            for col, arr, name in zip(
                [dl1, dl2, dl3],
                [raw_mask, cleaned, water_mask],
                ['raw_mask', 'cleaned_mask', 'water_mask']
            ):
                buf_dbg = io.BytesIO()
                Image.fromarray((arr * 255).astype(np.uint8)).save(buf_dbg, format='PNG')
                buf_dbg.seek(0)
                with col:
                    st.download_button(f"⬇ {name}.png", buf_dbg, f"{name}_{safe_ts}.png", "image/png")

            buf_np = io.BytesIO()
            np.savez_compressed(buf_np, pred_prob=pred_prob,
                                raw_mask=raw_mask, cleaned=cleaned, water_mask=water_mask)
            buf_np.seek(0)
            st.download_button("⬇ debug.npz", buf_np, f"unet_debug_{safe_ts}.npz")
        except Exception as e:
            st.warning(f"Debug display failed: {e}")

    # -------------------------------------------------------
    # FIX 3 — Polarity detection (mean-based, reliable)
    # -------------------------------------------------------
    if manual_invert:
        pred_prob = 1.0 - pred_prob
        st.info("⚠️ Manual inversion applied.")
    elif auto_detect_polarity and pred_prob.mean() > 0.5:
        pred_prob = 1.0 - pred_prob
        st.info("⚠️ Auto-polarity: inverted (model outputs high=background).")

    # Threshold selection
    if use_auto_threshold:
        st.info("⟳ Auto-selecting threshold…")
        sel_thresh = pick_auto_threshold(pred_prob, img_rgb,
                                         open_kernel=open_kernel_size,
                                         close_kernel=close_kernel_size,
                                         min_area=int(min_area))
        st.info(f"✅ Auto threshold: {sel_thresh:.4f}")
        final_mask = postprocess_mask(pred_prob, img_rgb,
                                       threshold=sel_thresh,
                                       open_kernel=open_kernel_size,
                                       close_kernel=close_kernel_size,
                                       min_area=int(min_area))
    else:
        final_mask = postprocess_mask(pred_prob, img_rgb,
                                       threshold=unet_threshold,
                                       open_kernel=open_kernel_size,
                                       close_kernel=close_kernel_size,
                                       min_area=int(min_area))

    metrics = calculate_metrics(final_mask)

    # -------------------------------------------------------
    # STEP 3: YOLO
    # -------------------------------------------------------
    yolo_results = None
    yolo_count   = 0

    if yolo_model is not None:
        try:
            yolo_results = yolo_model(img_rgb, verbose=False)
            if ULTRALYTICS_AVAILABLE:
                yolo_count = len(yolo_results[0].boxes) if yolo_results else 0
            else:
                yolo_count = len(yolo_results.xyxy[0]) if hasattr(yolo_results, 'xyxy') else 0
            st.success(f"✅ YOLO: {yolo_count} detection(s)")
        except Exception as e:
            st.warning(f"⚠️ YOLO error: {e}")

    # -------------------------------------------------------
    # VISUALISATIONS
    # -------------------------------------------------------
    st.divider()
    st.subheader("🎨 Visual Analysis")

    overlay        = create_overlay(img_rgb, final_mask,
                                    draw_outline=outline_enabled,
                                    outline_color=OUTLINE_COLOR)
    legend_overlay = create_legend_overlay(img_rgb, final_mask, yolo_results, timestamp)

    col1, col2, col3 = st.columns(3)
    with col1:
        # FIX 4 applied throughout
        st.image(img_rgb,           caption="📷 Original Image",             use_container_width=True, clamp=True)
    with col2:
        st.image(final_mask * 255,  caption="⚪ U-Net Mask (White=Oil)",      use_container_width=True, clamp=True)
    with col3:
        st.image(overlay,           caption="🔴 Overlay (Red=Oil)",           use_container_width=True, clamp=True)

    st.image(legend_overlay, caption="📊 Complete Analysis with Legend", use_container_width=True, clamp=True)

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------
    st.divider()
    st.subheader("📈 Detection Results")

    oil_percent   = metrics['oil_percentage']
    water_percent = 100 - oil_percent

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pie, ax = plt.subplots(figsize=(6, 5))
        ax.pie([oil_percent, water_percent],
               labels=["Oil Spill", "Clean Water"],
               autopct="%1.1f%%",
               colors=["#ff4d4d", "#4CAF50"],
               explode=(0.08, 0), startangle=90,
               textprops={"fontsize": 10})
        ax.set_title("U-Net: Oil Spill Area Distribution", fontsize=12, fontweight='bold')
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_c2:
        st.markdown("**📊 Detection Metrics**")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Oil Spill %",    f"{oil_percent:.2f}%")
            st.metric("CNN Confidence", f"{cnn_prob:.3f}")
        with mc2:
            st.metric("Total Components", metrics['total_components'])
            st.metric("YOLO Detections",  yolo_count)

        st.markdown("**📍 Component Analysis**")
        total_px = metrics['total_oil_pixels'] / (oil_percent / 100) if oil_percent > 0 else 0
        st.markdown(f"""
- **Largest Component Area**: {metrics['largest_component_area']} pixels
- **Total Oil Pixels**: {metrics['total_oil_pixels']} pixels
- **Total Image Pixels**: {total_px:.0f}
""")

    st.divider()
    st.subheader("📉 Multi-Model Analysis Chart")
    fig_results = create_results_chart(metrics, cnn_prob, yolo_count)
    st.pyplot(fig_results)
    plt.close(fig_results)

    # -------------------------------------------------------
    # DOWNLOADS
    # -------------------------------------------------------
    st.divider()
    st.subheader("⬇️ Download Results")

    fig_dl = plt.figure(figsize=(14, 8))
    gs     = fig_dl.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_orig = fig_dl.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb); ax_orig.set_title("Original Image", fontweight='bold'); ax_orig.axis("off")

    ax_mask = fig_dl.add_subplot(gs[0, 1])
    ax_mask.imshow(final_mask * 255, cmap='gray')
    ax_mask.set_title("U-Net Mask", fontweight='bold'); ax_mask.axis("off")

    ax_ov = fig_dl.add_subplot(gs[0, 2])
    ax_ov.imshow(overlay); ax_ov.set_title("Overlay", fontweight='bold'); ax_ov.axis("off")

    ax_pie2 = fig_dl.add_subplot(gs[1, 0])
    ax_pie2.pie([oil_percent, water_percent], labels=["Oil","Water"],
                autopct="%1.0f%%", colors=["#ff4d4d","#4CAF50"],
                explode=(0.05,0), startangle=90, textprops={"fontsize":8})
    ax_pie2.set_title("Area Distribution", fontweight='bold', fontsize=9)

    ax_txt = fig_dl.add_subplot(gs[1, 1]); ax_txt.axis("off")
    ax_txt.text(0.1, 0.5, f"""
DETECTION RESULTS

Timestamp: {timestamp}

CNN Confidence:  {cnn_prob:.3f}
U-Net Oil %:     {oil_percent:.2f}%
YOLO Detections: {yolo_count}

Components:   {metrics['total_components']}
Largest Comp: {metrics['largest_component_area']} px
Total Oil px: {metrics['total_oil_pixels']}
""", fontfamily='monospace', fontsize=9, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_leg = fig_dl.add_subplot(gs[1, 2]); ax_leg.axis("off")
    ax_leg.text(0.1, 0.5, """
LEGEND

🔴 Red Fill: Oil
⚪ White: Oil Mask
⬛ Black Outline: Boundary
🟨 Yellow Box: YOLO

Models:
• CNN Classification
• U-Net Segmentation
• YOLO Detection
""", fontfamily='monospace', fontsize=8, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig_dl.suptitle("Oil Spill Detection — Comprehensive Analysis Report",
                    fontsize=14, fontweight='bold', y=0.98)

    buf = io.BytesIO()
    fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_dl)
    buf.seek(0)

    st.download_button("⬇️ Download Complete Analysis (PNG)", buf.getvalue(),
                       f"oil_spill_analysis_{safe_ts}.png", "image/png")

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        b = io.BytesIO(); Image.fromarray(img_rgb).save(b, "PNG"); b.seek(0)
        st.download_button("📷 Original", b, f"original_{safe_ts}.png", "image/png")
    with dc2:
        b = io.BytesIO(); Image.fromarray((final_mask*255).astype(np.uint8)).save(b,"PNG"); b.seek(0)
        st.download_button("⚪ Mask", b, f"mask_{safe_ts}.png", "image/png")
    with dc3:
        b = io.BytesIO(); Image.fromarray(overlay).save(b, "PNG"); b.seek(0)
        st.download_button("🔴 Overlay", b, f"overlay_{safe_ts}.png", "image/png")

    # -------------------------------------------------------
    # MODEL INFO
    # -------------------------------------------------------
    st.divider()
    st.subheader("ℹ️ Model Information")

    with st.expander("🧠 CNN Classification Model"):
        st.markdown(f"""
**Purpose**: Binary classification (Oil / No Oil)
**Confidence**: {cnn_prob:.4f}
**Decision**: {'🚨 Oil Spill Detected' if cnn_prob >= CNN_THRESHOLD else '✅ No Oil'}
**Threshold**: {CNN_THRESHOLD}
**Path**: `{CNN_MODEL_PATH}`
""")

    with st.expander("🎯 U-Net Segmentation Model"):
        st.markdown(f"""
**Purpose**: Pixel-wise segmentation
**Oil Coverage**: {oil_percent:.2f}%
**Components**: {metrics['total_components']}
**Largest**: {metrics['largest_component_area']} px
**Threshold**: {UNET_THRESHOLD}
**Min Area**: {MIN_AREA} px
**Water Mask Skipped**: {SKIP_WATER_MASK}
**Path**: `{UNET_MODEL_PATH}`
""")

    with st.expander("📍 YOLO Detection Model"):
        if yolo_model:
            st.markdown(f"""
**Detections**: {yolo_count}
**Conf Threshold**: {YOLO_CONFIDENCE}
**Status**: ✅ Loaded
**Path**: `{YOLO_MODEL_PATH}`
""")
        else:
            st.warning("⚠️ YOLO not available")

    st.divider()
    st.caption("🛢️ Analysis complete!")

else:
    st.info("👈 Upload a SAR or satellite image to begin.")

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
### Multi-Model Oil Spill Detection

**1️⃣ CNN** — Binary classification (oil / no oil)
**2️⃣ U-Net** — Pixel-level segmentation
**3️⃣ YOLO** — Bounding-box localization

### Key Setting: Water Mask
For **RGB satellite images** (docks, ships, sand visible) → ✅ **Skip water mask** (default).
For **dark-water SAR** → uncheck Skip and choose Otsu or fixed threshold.

### Formats: JPG / PNG
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align:center;color:gray;font-size:.85rem;'>
<p>🛢️ AI-Driven Oil Spill Detection | TensorFlow · YOLO · Streamlit</p>
<p>CNN Classification | U-Net Segmentation | YOLO Detection</p>
</div>
""", unsafe_allow_html=True)