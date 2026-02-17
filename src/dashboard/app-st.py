#!/usr/bin/env python3
"""
AI-Driven Oil Spill Detection  —  Final Production Version
Models : CNN (binary classification) · U-Net (segmentation) · YOLO (localisation)
Author : (your name)
"""

# =================================================
# IMPORTS
# =================================================
import io
import os
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =================================================
# CONSTANTS
# =================================================
IMG_SIZE       = 256
CNN_THRESHOLD  = 0.5
UNET_THRESHOLD = 0.4
MIN_AREA       = 600
YOLO_CONF      = 0.5
OPEN_KERNEL    = 5
CLOSE_KERNEL   = 5

# ── CHANGE 1 of 2 ─────────────────────────────────────────────────────────────
# Update these paths to match your machine.
# YOLO_REPO_PATH must point to the yolov5/ folder already inside your project.
# ──────────────────────────────────────────────────────────────────────────────
CNN_PATH       = Path(r"D:\Coding\SEM-8-NEW\OIL-SPILL\models\cnn\cnn_classifier.keras")
UNET_PATH      = Path(r"D:\Coding\SEM-8-NEW\OIL-SPILL\models\unet\unet_segmentation.keras")
YOLO_PATH      = Path(r"D:\Coding\SEM-8-NEW\OIL-SPILL\models\yolo\best.pt")
YOLO_REPO_PATH = Path(r"D:\Coding\SEM-8-NEW\OIL-SPILL\yolov5")

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="AI Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================
# CSS
# =================================================
st.markdown("""
<style>
.hero {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.2rem 1rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1  { margin: 0; font-size: 2.4rem; letter-spacing: -0.5px; }
.hero p   { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.75; }
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# MODEL LOADING  (cached — runs once per session)
# =================================================
@st.cache_resource
def load_cnn():
    try:
        m = tf.keras.models.load_model(str(CNN_PATH))
        return m, True
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_unet():
    try:
        m = tf.keras.models.load_model(str(UNET_PATH), compile=False)
        return m, True
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    # ── CHANGE 2 of 2 ───────────────────────────────────────────────────────
    # WHY THIS FIX IS NEEDED:
    #   Your best.pt was trained in Google Colab (Linux).
    #   Colab saves path objects as PosixPath (Linux format) inside the file.
    #   When PyTorch unpickles the file on Windows, it crashes:
    #     "cannot instantiate 'PosixPath' on your system"
    #
    # THE FIX:
    #   Before loading, temporarily remap PosixPath → WindowsPath.
    #   The finally block restores pathlib immediately after, no side effects.
    #   This only activates on Windows — does nothing on Linux/Mac.
    # ────────────────────────────────────────────────────────────────────────
    if not TORCH_AVAILABLE:
        return None, "torch not installed"

    yolo_abs = str(os.path.abspath(YOLO_PATH))
    if not os.path.exists(yolo_abs):
        return None, f"weights not found: {yolo_abs}"

    try:
        import pathlib
        import platform

        _patch_target = None
        if platform.system() == "Windows":
            _patch_target = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            repo_abs = str(os.path.abspath(YOLO_REPO_PATH))

            # Disable YOLOv5's internal requirements auto-checker.
            # It has a pip quoting bug on Python 3.12 that causes harmless
            # but noisy "InvalidMarker" warnings in the terminal.
            os.environ["YOLOV5_NO_CHECK_REQUIREMENTS"] = "1"

            if os.path.isdir(repo_abs):
                # Use the yolov5/ folder already in your project — no internet needed
                m = torch.hub.load(
                    repo_abs,
                    "custom",
                    path=yolo_abs,
                    source="local",
                    verbose=False,
                )
            else:
                # Fallback: download from GitHub (needs internet, runs once only)
                m = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=yolo_abs,
                    force_reload=False,
                    verbose=False,
                )
        finally:
            # Always restore pathlib — even if loading raised an exception
            if _patch_target is not None:
                pathlib.PosixPath = _patch_target

        m.conf = YOLO_CONF
        return m, True

    except Exception as e:
        return None, str(e)

cnn_model,  cnn_ok   = load_cnn()
unet_model, unet_ok  = load_unet()
yolo_model, yolo_ok  = load_yolo()

# =================================================
# CORE PROCESSING FUNCTIONS
# =================================================
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)


def postprocess_mask(pred_prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    raw = (pred_prob > threshold).astype(np.uint8)
    k_o = np.ones((OPEN_KERNEL,  OPEN_KERNEL),  np.uint8)
    k_c = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    cleaned = cv2.morphologyEx(raw,     cv2.MORPH_OPEN,  k_o)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_c)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    final = np.zeros_like(cleaned)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            final[labels == i] = 1
    return final.astype(np.uint8)


def auto_invert(pred_prob: np.ndarray) -> tuple[np.ndarray, bool]:
    if pred_prob.mean() > 0.5:
        return 1.0 - pred_prob, True
    return pred_prob, False


def build_overlay(img_rgb: np.ndarray, mask: np.ndarray,
                  alpha: float = 0.4, outline: bool = True) -> np.ndarray:
    img     = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    blended = img.copy().astype(np.float32)
    blended[mask == 1] = blended[mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if outline and mask.any():
        m8  = (mask * 255).astype(np.uint8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m8  = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, ker)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            eps    = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(blended, [approx], 0, (0, 0, 0), 2)
    return blended


def draw_yolo_boxes(base_img: np.ndarray, results, class_names: dict = None) -> np.ndarray:
    out = base_img.copy()
    if results is None:
        return out
    try:
        detections = results.xyxy[0].cpu().numpy()
        for det in detections:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            conf   = float(det[4])
            cls_id = int(det[5])
            label  = class_names.get(cls_id, f"cls{cls_id}") if class_names else f"cls{cls_id}"
            text   = f"{label} {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1, th + 8)), (0, 255, 255), -1)
            cv2.putText(out, text, (x1 + 2, max(y1 - 4, th + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass
    return out


def compute_metrics(mask: np.ndarray) -> dict:
    oil_px  = int(mask.sum())
    total   = int(mask.size)
    oil_pct = (oil_px / total * 100) if total else 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 0 if n == 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
    return {"oil_pct": oil_pct, "n_comps": n - 1,
            "largest_px": largest, "oil_px": oil_px, "total_px": total}

# =================================================
# CHART HELPERS
# =================================================
PIE_COLORS = ["#e63946", "#4CAF50"]

def pie_chart(oil_pct: float, title: str, figsize=(5, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie([oil_pct, 100 - oil_pct], labels=["Oil Spill", "Clean Water"],
           autopct="%1.1f%%", colors=PIE_COLORS, explode=(0.07, 0),
           startangle=90, textprops={"fontsize": 9})
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def summary_chart(metrics: dict, cnn_prob: float, yolo_count: int):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Multi-Model Detection Summary", fontsize=13, fontweight="bold")
    axes[0].bar(["CNN"], [cnn_prob * 100], color="#2196F3", alpha=0.8, width=0.4)
    axes[0].axhline(CNN_THRESHOLD * 100, color="red", ls="--", lw=1.5, label=f"Threshold {CNN_THRESHOLD}")
    axes[0].set_ylim(0, 100); axes[0].set_ylabel("Confidence (%)"); axes[0].legend(fontsize=8)
    axes[0].set_title("CNN Classification", fontweight="bold"); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(["Components", "Largest (px/10)"],
                [metrics["n_comps"], metrics["largest_px"] / 10],
                color=["#9C27B0", "#FF9800"], alpha=0.8)
    axes[1].set_title("U-Net Components", fontweight="bold"); axes[1].grid(axis="y", alpha=0.3)
    scores = [min(cnn_prob * 100, 100), metrics["oil_pct"], min(yolo_count * 20, 100)]
    axes[2].bar(["CNN\nConf %", "U-Net\nSpill %", f"YOLO\n({yolo_count} det)"],
                scores, color=["#2196F3", "#e63946", "#FF9800"], alpha=0.8)
    axes[2].set_ylim(0, 100); axes[2].set_title("All Models", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def report_figure(img_rgb, mask, overlay, metrics, cnn_prob, yolo_count, timestamp):
    oil_pct = metrics["oil_pct"]
    fig = plt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    ax_o = fig.add_subplot(gs[0, 0]); ax_o.imshow(img_rgb)
    ax_o.set_title("Original Image", fontweight="bold"); ax_o.axis("off")
    ax_m = fig.add_subplot(gs[0, 1]); ax_m.imshow(mask * 255, cmap="gray")
    ax_m.set_title("U-Net Segmentation Mask", fontweight="bold"); ax_m.axis("off")
    ax_v = fig.add_subplot(gs[0, 2]); ax_v.imshow(overlay)
    ax_v.set_title("Oil Spill Overlay", fontweight="bold"); ax_v.axis("off")
    ax_p = fig.add_subplot(gs[1, 0])
    ax_p.pie([oil_pct, 100 - oil_pct], labels=["Oil", "Water"],
             autopct="%1.0f%%", colors=PIE_COLORS,
             explode=(0.05, 0), startangle=90, textprops={"fontsize": 8})
    ax_p.set_title("Area Distribution", fontweight="bold", fontsize=9)
    ax_t = fig.add_subplot(gs[1, 1]); ax_t.axis("off")
    ax_t.text(0.08, 0.5, (
        f"DETECTION RESULTS\n\n"
        f"Timestamp : {timestamp}\n\n"
        f"CNN Confidence  : {cnn_prob:.3f}\n"
        f"U-Net Oil Cover : {oil_pct:.2f}%\n"
        f"YOLO Detections : {yolo_count}\n\n"
        f"Components      : {metrics['n_comps']}\n"
        f"Largest Region  : {metrics['largest_px']} px\n"
        f"Total Oil Pixels: {metrics['oil_px']}"
    ), fontfamily="monospace", fontsize=8.5, va="center",
       bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    ax_l = fig.add_subplot(gs[1, 2]); ax_l.axis("off")
    ax_l.text(0.08, 0.5, (
        "LEGEND\n\n"
        "🔴  Red fill     → Oil region\n"
        "⚪  White mask  → Segmentation\n"
        "⬛  Black line   → Oil boundary\n"
        "🟨  Cyan box    → YOLO detection\n\n"
        "MODELS\n"
        "• CNN  — binary classifier\n"
        "• U-Net — pixel segmentation\n"
        "• YOLO  — bounding-box detect"
    ), fontfamily="monospace", fontsize=8.5, va="center",
       bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.6))
    fig.suptitle("Oil Spill Detection — Comprehensive Analysis Report",
                 fontsize=14, fontweight="bold", y=0.99)
    return fig


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def img_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# =================================================
# HEADER
# =================================================
st.markdown("""
<div class="hero">
    <h1>🛢️ AI-Driven Oil Spill Detection</h1>
    <p>CNN · U-Net · YOLO — multi-model satellite image analysis</p>
</div>
""", unsafe_allow_html=True)

timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
st.caption(f"⏰ Session timestamp: {timestamp}")
st.divider()

# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("<p class='section-label'>Model Status</p>", unsafe_allow_html=True)
    st.markdown(f"{'✅' if cnn_ok  is True else '❌'} **CNN**  `{CNN_PATH.name}`")
    st.markdown(f"{'✅' if unet_ok is True else '❌'} **U-Net** `{UNET_PATH.name}`")
    st.markdown(f"{'✅' if yolo_ok is True else '⚠️'} **YOLO** `{YOLO_PATH.name}`")
    st.divider()
    st.markdown("<p class='section-label'>Detection Sensitivity</p>", unsafe_allow_html=True)
    cnn_thr  = st.slider("CNN threshold",          0.0, 1.0, CNN_THRESHOLD,  0.01,
                          help="Minimum CNN confidence to trigger segmentation")
    unet_thr = st.slider("Segmentation threshold", 0.0, 1.0, UNET_THRESHOLD, 0.01,
                          help="Lower = more oil detected · Higher = stricter")
    min_area = st.number_input("Min spill size (px)", 0, value=MIN_AREA, step=50,
                                help="Blobs smaller than this are ignored")
    yolo_cf  = st.slider("YOLO confidence",         0.0, 1.0, YOLO_CONF,     0.01,
                          help="Minimum YOLO bounding-box confidence")
    st.divider()
    st.markdown("<p class='section-label'>Visualisation</p>", unsafe_allow_html=True)
    show_outline = st.toggle("Oil boundary outline", value=True)
    st.divider()
    st.markdown("<p class='section-label'>Colour Guide</p>", unsafe_allow_html=True)
    st.markdown("""
🟥 **Red fill** — oil region  
⬛ **Black line** — oil boundary  
🟨 **Cyan box** — YOLO detection  
⬜ **White mask** — segmentation output  
    """)

# =================================================
# UPLOAD
# =================================================
st.subheader("📤 Upload Image")
uploaded = st.file_uploader(
    "SAR or satellite image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

# =================================================
# WELCOME SCREEN
# =================================================
if not uploaded:
    st.info("Upload an image above to begin analysis.")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Step 1 — CNN classifier**  
Checks whether the image contains an oil spill at all.
If confidence is below the threshold, analysis stops here.

**Step 2 — U-Net segmentation**  
Produces a pixel-level mask highlighting the exact oil region.

**Step 3 — YOLO detection** *(if model is loaded)*  
Draws bounding boxes around detected oil patches for quick localisation.

**Outputs**
- Side-by-side: original · mask · overlay
- Metrics: coverage %, component count, CNN score
- Downloadable report PNG + individual images
        """)
    st.stop()

# =================================================
# INFERENCE
# =================================================
if cnn_model is None or unet_model is None:
    st.error("❌ CNN or U-Net model failed to load — check model paths.")
    st.stop()

raw     = np.frombuffer(uploaded.read(), np.uint8)
img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
safe_ts = timestamp.replace(":", "-").replace(" ", "_")

st.image(img_rgb, caption="Uploaded image", width="stretch")

with st.spinner("Running models…"):

    batch    = preprocess(img_rgb)
    cnn_prob = float(cnn_model.predict(batch, verbose=0)[0][0])

    if cnn_prob < cnn_thr:
        st.success(f"✅ No oil spill detected  (CNN score: {cnn_prob:.3f} < threshold {cnn_thr})")
        st.stop()

    st.warning(f"🚨 Oil spill detected  (CNN score: {cnn_prob:.3f})")

    pred_prob = unet_model.predict(batch, verbose=0)[0].squeeze()
    pred_prob, inverted = auto_invert(pred_prob)
    if inverted:
        st.info("ℹ️ U-Net polarity auto-corrected (model outputs high = background).")

    final_mask = postprocess_mask(pred_prob, unet_thr, int(min_area))
    metrics    = compute_metrics(final_mask)

    yolo_results = None
    yolo_count   = 0
    if yolo_model is not None:
        try:
            if yolo_model.conf != yolo_cf:
                yolo_model.conf = yolo_cf
            yolo_results = yolo_model(img_rgb)
            yolo_count   = len(yolo_results.xyxy[0]) if yolo_results is not None else 0
        except Exception as e:
            st.warning(f"⚠️ YOLO inference failed: {e}")

    overlay      = build_overlay(img_rgb, final_mask, outline=show_outline)
    yolo_names   = yolo_model.names if yolo_model is not None else {}
    overlay_yolo = draw_yolo_boxes(overlay, yolo_results, class_names=yolo_names)

# =================================================
# RESULTS — VISUAL
# =================================================
st.divider()
st.subheader("🎨 Visual Analysis")

c1, c2, c3 = st.columns(3)
with c1:
    st.image(img_rgb,          caption="📷 Original",      width="stretch")
with c2:
    st.image(final_mask * 255, caption="⚪ U-Net Mask",     width="stretch", clamp=True)
with c3:
    st.image(overlay_yolo,     caption="🔴 Overlay + YOLO", width="stretch")

# =================================================
# RESULTS — METRICS
# =================================================
st.divider()
st.subheader("📊 Detection Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Oil Coverage",    f"{metrics['oil_pct']:.2f}%")
m2.metric("CNN Score",       f"{cnn_prob:.3f}")
m3.metric("Spill Regions",   metrics["n_comps"])
m4.metric("YOLO Detections", yolo_count)

col_pie, col_chart = st.columns([1, 2])
with col_pie:
    st.pyplot(pie_chart(metrics["oil_pct"], "Oil vs Clean Water"))
    plt.close()
with col_chart:
    st.pyplot(summary_chart(metrics, cnn_prob, yolo_count))
    plt.close()

# =================================================
# RESULTS — MODEL DETAIL
# =================================================
st.divider()
st.subheader("ℹ️ Model Details")

with st.expander("🧠 CNN — Binary Classifier"):
    st.markdown(f"""
| Field | Value |
|---|---|
| Confidence score | `{cnn_prob:.4f}` |
| Threshold | `{cnn_thr}` |
| Decision | {'🚨 Oil detected' if cnn_prob >= cnn_thr else '✅ Clean'} |
| Model file | `{CNN_PATH.name}` |
""")

with st.expander("🎯 U-Net — Segmentation"):
    st.markdown(f"""
| Field | Value |
|---|---|
| Oil coverage | `{metrics['oil_pct']:.2f}%` |
| Spill regions | `{metrics['n_comps']}` |
| Largest region | `{metrics['largest_px']} px` |
| Total oil pixels | `{metrics['oil_px']}` |
| Threshold used | `{unet_thr}` |
| Min blob size | `{int(min_area)} px` |
| Polarity inverted | `{inverted}` |
| Model file | `{UNET_PATH.name}` |
""")

with st.expander("📍 YOLO — Object Detection"):
    if yolo_model:
        st.markdown(f"""
| Field | Value |
|---|---|
| Detections | `{yolo_count}` |
| Confidence threshold | `{yolo_cf}` |
| Status | ✅ Loaded |
| Model file | `{YOLO_PATH.name}` |
""")
    else:
        st.warning(f"⚠️ YOLO unavailable — {yolo_ok}")

# =================================================
# DOWNLOADS
# =================================================
st.divider()
st.subheader("⬇️ Download Results")

report_png = fig_to_bytes(
    report_figure(img_rgb, final_mask, overlay_yolo, metrics, cnn_prob, yolo_count, timestamp)
)

st.download_button(
    "⬇️ Full Analysis Report (PNG)",
    report_png,
    f"oil_spill_report_{safe_ts}.png",
    "image/png",
    use_container_width=True,
)

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("📷 Original", img_to_bytes(img_rgb),
                       f"original_{safe_ts}.png", "image/png", use_container_width=True)
with d2:
    st.download_button("⚪ Mask", img_to_bytes((final_mask * 255).astype(np.uint8)),
                       f"mask_{safe_ts}.png", "image/png", use_container_width=True)
with d3:
    st.download_button("🔴 Overlay", img_to_bytes(overlay_yolo),
                       f"overlay_{safe_ts}.png", "image/png", use_container_width=True)

# =================================================
# FOOTER
# =================================================
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:.82rem;'>
    🛢️ AI-Driven Oil Spill Detection &nbsp;·&nbsp;
    TensorFlow &nbsp;·&nbsp; YOLO &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)