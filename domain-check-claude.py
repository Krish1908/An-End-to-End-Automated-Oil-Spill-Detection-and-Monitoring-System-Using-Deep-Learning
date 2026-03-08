"""
=============================================================================
DOMAIN SHIFT ANALYSIS
Dataset 3: Zenodo 15298010  — Refined Deep-SAR SOS Dataset
Dataset 4: CSIRO via Kaggle — Sentinel-1 SAR Oil Spill Dataset
=============================================================================
Analyzes domain shift across:
 - Image statistics (mean, std, min, max, dynamic range)
 - Color / channel distributions (histograms)
 - Texture features (GLCM contrast, homogeneity, energy, correlation)
 - Frequency domain (FFT energy distribution)
 - Mask/annotation statistics (class ratios, spill coverage)
 - MMD (Maximum Mean Discrepancy) distance between domains
 - t-SNE / PCA feature visualization
 - Per-subfolder / per-split breakdown
 - Channel distribution analysis (1ch vs 3ch detection)
=============================================================================
Usage:
    python domain_shift_analysis.py
    (Adjust DATASET3_PATH and DATASET4_PATH at the top if needed)
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit paths if needed
# ─────────────────────────────────────────────────────────────────────────────
DATASET3_PATH = r"D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_3"
DATASET4_PATH = r"D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_4"
OUTPUT_DIR    = r"D:\Coding\SEM-8-NEW\OIL-SPILL\domain_shift_results_3_4"

MAX_IMAGES_PER_DATASET = 300
MAX_HIST_IMAGES        = 500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS

def is_mask(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name  = path.stem.lower()
    keywords = ("annot", "mask", "label", "seg", "gt", "ground")
    return any(k in p for k in keywords for p in parts + [name])

def collect_files(root: str):
    root = Path(root)
    images, masks = [], []
    structure = defaultdict(lambda: {"images": 0, "masks": 0})
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not is_image(p):
            continue
        if "__MACOSX" in p.parts:
            continue
        rel = str(p.parent.relative_to(root))
        if is_mask(p):
            masks.append(p)
            structure[rel]["masks"] += 1
        else:
            images.append(p)
            structure[rel]["images"] += 1
    return images, masks, dict(structure)


def load_original_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def image_stats(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    true_channels = 1 if len(img.shape) == 2 else img.shape[2]
    if len(img.shape) == 2:
        gray = img.astype(np.float32)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.max() > 255:
        gray = (gray / gray.max() * 255)
    channels = cv2.split(img) if len(img.shape) == 3 else [img]
    h, w = gray.shape
    return {
        "height": h,
        "width":  w,
        "channels": true_channels,
        "mean":   float(gray.mean()),
        "std":    float(gray.std()),
        "min":    float(gray.min()),
        "max":    float(gray.max()),
        "dynamic_range": float(gray.max() - gray.min()),
        "mean_r": float(channels[2].mean()) if len(channels) >= 3 else float(gray.mean()),
        "mean_g": float(channels[1].mean()) if len(channels) >= 3 else float(gray.mean()),
        "mean_b": float(channels[0].mean()) if len(channels) >= 3 else float(gray.mean()),
    }


def glcm_features(gray_img):
    g = img_as_ubyte(gray_img / (gray_img.max() + 1e-8))
    glcm = graycomatrix(g, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    return {
        "contrast":      float(graycoprops(glcm, "contrast").mean()),
        "homogeneity":   float(graycoprops(glcm, "homogeneity").mean()),
        "energy":        float(graycoprops(glcm, "energy").mean()),
        "correlation":   float(graycoprops(glcm, "correlation").mean()),
        "dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
    }


def fft_features(gray_img):
    f      = np.fft.fft2(gray_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag    = np.abs(fshift)
    h, w   = mag.shape
    cy, cx = h // 2, w // 2
    r      = min(h, w) // 8
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    low    = mag[dist <= r].sum()
    high   = mag[dist >  r].sum()
    total  = low + high + 1e-8
    return {
        "fft_low_ratio":    float(low / total),
        "fft_high_ratio":   float(high / total),
        "fft_energy_total": float(total),
    }


def mmd_rbf(X, Y, gamma=1.0):
    from sklearn.metrics.pairwise import rbf_kernel
    n, m = len(X), len(Y)
    Kxx  = rbf_kernel(X, X, gamma)
    Kyy  = rbf_kernel(Y, Y, gamma)
    Kxy  = rbf_kernel(X, Y, gamma)
    return Kxx.sum()/(n*n) + Kyy.sum()/(m*m) - 2*Kxy.sum()/(n*m)


def aggregate_histogram(paths, n=MAX_HIST_IMAGES):
    hist_agg = [np.zeros(256), np.zeros(256), np.zeros(256)]
    for p in paths[:n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        for c in range(3):
            h, _ = np.histogram(img[:, :, c], bins=256, range=(0, 256))
            hist_agg[c] += h
    for c in range(3):
        s = hist_agg[c].sum()
        if s > 0:
            hist_agg[c] /= s
    return hist_agg


def extract_flat_features(paths, n=MAX_IMAGES_PER_DATASET, size=(64, 64)):
    """Extract flattened grayscale features for PCA / t-SNE / MMD."""
    feats = []
    for p in tqdm(paths[:n], desc=f"  Feature extraction ({min(n,len(paths))} imgs)", leave=False):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.resize(gray, size).astype(np.float32) / 255.0
        feats.append(gray.flatten())
    return np.array(feats)   # FIX: was missing — caused crash at StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  OIL SPILL DOMAIN SHIFT ANALYSIS")
    print("=" * 70)

    # ── 1. Discover files ────────────────────────────────────────────────────
    print("\n[1/8] Scanning dataset directories …")
    imgs1, masks1, struct1 = collect_files(DATASET3_PATH)
    imgs2, masks2, struct2 = collect_files(DATASET4_PATH)

    print(f"\n  Dataset 3: {len(imgs1)} images,  {len(masks1)} masks")
    print(f"  Dataset 4: {len(imgs2)} images,  {len(masks2)} masks")

    print("\n  ── Dataset 3 folder structure ──")
    for folder, counts in sorted(struct1.items()):
        print(f"    {folder:50s}  images={counts['images']:4d}  masks={counts['masks']:4d}")

    print("\n  ── Dataset 4 folder structure ──")
    for folder, counts in sorted(struct2.items()):
        print(f"    {folder:50s}  images={counts['images']:4d}  masks={counts['masks']:4d}")

    # ── 2. Per-image statistics ───────────────────────────────────────────────
    print("\n[2/8] Computing per-image pixel statistics …")

    def gather_stats(paths, label):
        all_stats = []
        for p in tqdm(paths, desc=f"  Stats {label}", leave=False):
            s = image_stats(p)
            if s:
                all_stats.append(s)
        return all_stats

    stats1 = gather_stats(imgs1, "D3")
    stats2 = gather_stats(imgs2, "D4")

    def summarize(stats_list, key):
        vals = [s[key] for s in stats_list if key in s]
        return {"mean": np.mean(vals), "std": np.std(vals),
                "min":  np.min(vals),  "max": np.max(vals)} if vals else {}

    keys_to_compare = ["mean", "std", "dynamic_range",
                       "mean_r", "mean_g", "mean_b", "height", "width"]
    stat_summary = {}
    for k in keys_to_compare:
        stat_summary[k] = {
            "D3": summarize(stats1, k),
            "D4": summarize(stats2, k),
        }

    # ── 3. Texture features ──────────────────────────────────────────────────
    print("\n[3/8] Extracting GLCM texture features …")

    def gather_texture(paths, n, label):
        feats = []
        for p in tqdm(paths[:n], desc=f"  Texture {label}", leave=False):
            g = load_original_gray(p)
            if g is not None:
                feats.append(glcm_features(g))
        return feats

    tex1 = gather_texture(imgs1, MAX_IMAGES_PER_DATASET, "D3")
    tex2 = gather_texture(imgs2, MAX_IMAGES_PER_DATASET, "D4")

    tex_keys = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
    tex_summary = {}
    for k in tex_keys:
        v1 = [t[k] for t in tex1]
        v2 = [t[k] for t in tex2]
        wd       = wasserstein_distance(v1, v2) if v1 and v2 else 0
        ks_s, kp = ks_2samp(v1, v2)             if v1 and v2 else (0, 1)
        tex_summary[k] = {
            "D3_mean": np.mean(v1), "D3_std": np.std(v1),
            "D4_mean": np.mean(v2), "D4_std": np.std(v2),
            "wasserstein": wd, "ks_stat": ks_s, "ks_p": kp,
        }

    # ── 4. FFT features ──────────────────────────────────────────────────────
    print("\n[4/8] FFT frequency-domain analysis …")

    def gather_fft(paths, n, label):
        feats = []
        for p in tqdm(paths[:n], desc=f"  FFT {label}", leave=False):
            g = load_original_gray(p)
            if g is not None:
                feats.append(fft_features(g))
        return feats

    fft1 = gather_fft(imgs1, MAX_IMAGES_PER_DATASET, "D3")
    fft2 = gather_fft(imgs2, MAX_IMAGES_PER_DATASET, "D4")

    fft_summary = {}
    for k in ["fft_low_ratio", "fft_high_ratio"]:
        v1 = [f[k] for f in fft1]
        v2 = [f[k] for f in fft2]
        fft_summary[k] = {
            "D3_mean":    np.mean(v1) if v1 else 0,
            "D4_mean":    np.mean(v2) if v2 else 0,
            "wasserstein": wasserstein_distance(v1, v2) if v1 and v2 else 0,
        }

    # ── 5. Histograms ────────────────────────────────────────────────────────
    print("\n[5/8] Aggregating channel histograms …")
    hist1 = aggregate_histogram(imgs1)
    hist2 = aggregate_histogram(imgs2)
    hist_wasserstein = [wasserstein_distance(hist1[c], hist2[c]) for c in range(3)]

    # ── 6. MMD ───────────────────────────────────────────────────────────────
    print("\n[6/8] Computing MMD (Maximum Mean Discrepancy) …")
    print("  Extracting flat features for Dataset 3 …")
    F1 = extract_flat_features(imgs1)
    print("  Extracting flat features for Dataset 4 …")
    F2 = extract_flat_features(imgs2)

    scaler = StandardScaler()
    F1_s   = scaler.fit_transform(F1)
    F2_s   = scaler.transform(F2)

    pca50    = PCA(n_components=50, random_state=42)
    pca50.fit(np.vstack([F1_s, F2_s]))
    F1_pca   = pca50.transform(F1_s)
    F2_pca   = pca50.transform(F2_s)

    mmd_val  = mmd_rbf(F1_pca, F2_pca)
    print(f"  MMD² = {mmd_val:.6f}")

    # ── 7. PCA + t-SNE visualization ─────────────────────────────────────────
    print("\n[7/8] PCA & t-SNE visualization …")
    pca2         = PCA(n_components=2, random_state=42)
    combined_pca = pca2.fit_transform(np.vstack([F1_pca, F2_pca]))
    n1           = len(F1_pca)

    tsne_n = min(500, n1, len(F2_pca))
    rng    = np.random.default_rng(42)
    idx1   = rng.choice(n1,          tsne_n, replace=False)
    idx2   = rng.choice(len(F2_pca), tsne_n, replace=False)
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_out = tsne.fit_transform(np.vstack([F1_pca[idx1], F2_pca[idx2]]))

    # ── 8. Mask statistics ───────────────────────────────────────────────────
    print("\n[8/8] Analysing mask/annotation coverage …")

    def mask_coverage(mask_paths, n, label):
        ratios = []
        for p in tqdm(mask_paths[:n], desc=f"  Masks {label}", leave=False):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = cv2.imread(str(p))
                if m is None:
                    continue
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            ratios.append(float((m > 0).mean()))
        return ratios

    cov1 = mask_coverage(masks1, MAX_IMAGES_PER_DATASET, "D3")
    cov2 = mask_coverage(masks2, MAX_IMAGES_PER_DATASET, "D4")

    # ═════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\nGenerating plots …")
    colors = {"D3": "#2196F3", "D4": "#FF5722"}

    # Figure 1 — Pixel statistics
    fig1, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig1.suptitle("Dataset Overview & Pixel Statistics", fontsize=16, fontweight="bold")
    stat_plot_keys = ["mean", "std", "dynamic_range", "mean_r", "mean_g", "mean_b", "height", "width"]
    titles = ["Pixel Mean", "Pixel Std Dev", "Dynamic Range",
              "Mean Red Channel", "Mean Green Channel", "Mean Blue Channel",
              "Image Height (px)", "Image Width (px)"]
    for ax, k, title in zip(axes.flatten(), stat_plot_keys, titles):
        d1_vals = [s[k] for s in stats1 if k in s]
        d2_vals = [s[k] for s in stats2 if k in s]
        ax.hist(d1_vals, bins=40, alpha=0.65, color=colors["D3"], label="D3: Zenodo SAR", density=True)
        ax.hist(d2_vals, bins=40, alpha=0.65, color=colors["D4"], label="D4: CSIRO SAR",  density=True)
        ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
        if d1_vals and d2_vals:
            wd = wasserstein_distance(d1_vals, d2_vals)
            ax.set_title(f"{title}\nWasserstein={wd:.2f}", fontsize=9)
        else:
            ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "1_pixel_statistics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1); print("  Saved: 1_pixel_statistics.png")

    # Figure 2 — Channel histograms
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Aggregated Channel Intensity Histograms", fontsize=14, fontweight="bold")
    for ax, c, cname in zip(axes, range(3), ["Red / B0", "Green / B1", "Blue / B2"]):
        ax.plot(hist1[c], color=colors["D3"], lw=1.8, label="D3: Zenodo SAR")
        ax.plot(hist2[c], color=colors["D4"], lw=1.8, label="D4: CSIRO SAR")
        ax.set_title(f"Channel: {cname}\nWasserstein={hist_wasserstein[c]:.4f}")
        ax.set_xlabel("Pixel Intensity"); ax.set_ylabel("Normalized Frequency"); ax.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "2_channel_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2); print("  Saved: 2_channel_histograms.png")

    # Figure 3 — Texture
    fig3, axes = plt.subplots(1, len(tex_keys), figsize=(20, 5))
    fig3.suptitle("GLCM Texture Feature Distributions", fontsize=14, fontweight="bold")
    for ax, k in zip(axes, tex_keys):
        v1 = [t[k] for t in tex1]; v2 = [t[k] for t in tex2]
        ax.hist(v1, bins=40, alpha=0.65, color=colors["D3"], label="D3: Zenodo SAR", density=True)
        ax.hist(v2, bins=40, alpha=0.65, color=colors["D4"], label="D4: CSIRO SAR",  density=True)
        wd = tex_summary[k]["wasserstein"]; ks = tex_summary[k]["ks_stat"]
        ax.set_title(f"{k.capitalize()}\nWasserstein={wd:.4f} | KS={ks:.3f}", fontsize=9)
        ax.set_xlabel(k); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "3_texture_features.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3); print("  Saved: 3_texture_features.png")

    # Figure 4 — FFT
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle("Frequency Domain Analysis (FFT)", fontsize=14, fontweight="bold")
    for ax, k in zip(axes, ["fft_low_ratio", "fft_high_ratio"]):
        v1 = [f[k] for f in fft1]; v2 = [f[k] for f in fft2]
        ax.hist(v1, bins=40, alpha=0.65, color=colors["D3"], label="D3: Zenodo SAR", density=True)
        ax.hist(v2, bins=40, alpha=0.65, color=colors["D4"], label="D4: CSIRO SAR",  density=True)
        wd = fft_summary[k]["wasserstein"]
        ax.set_title(f"{k}\nWasserstein={wd:.4f}", fontsize=10)
        ax.set_xlabel("Ratio"); ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "4_fft_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4); print("  Saved: 4_fft_analysis.png")

    # Figure 5 — PCA + t-SNE
    fig5, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig5.suptitle(f"Feature Space Visualization  (MMD² = {mmd_val:.5f})", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.scatter(combined_pca[:n1, 0], combined_pca[:n1, 1], c=colors["D3"], alpha=0.4, s=15, label="D3: Zenodo SAR")
    ax.scatter(combined_pca[n1:, 0], combined_pca[n1:, 1], c=colors["D4"], alpha=0.4, s=15, label="D4: CSIRO SAR")
    ax.set_title("PCA (2 components)", fontsize=12); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    ax = axes[1]
    ax.scatter(tsne_out[:tsne_n, 0], tsne_out[:tsne_n, 1], c=colors["D3"], alpha=0.4, s=15, label="D3: Zenodo SAR")
    ax.scatter(tsne_out[tsne_n:, 0], tsne_out[tsne_n:, 1], c=colors["D4"], alpha=0.4, s=15, label="D4: CSIRO SAR")
    ax.set_title("t-SNE (perplexity=30)", fontsize=12); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.legend()
    plt.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_DIR, "5_pca_tsne.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5); print("  Saved: 5_pca_tsne.png")

    # Figure 6 — Mask coverage
    fig6, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig6.suptitle("Annotation / Mask Coverage (Spill Region Ratio)", fontsize=14, fontweight="bold")
    for ax, vals, label, color in [
        (axes[0], cov1, "D3: Zenodo SAR", colors["D3"]),
        (axes[1], cov2, "D4: CSIRO SAR",  colors["D4"]),
    ]:
        if vals:
            ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(vals), color="black", linestyle="--", label=f"Mean={np.mean(vals):.3f}")
            ax.set_title(f"{label}\n(n={len(vals)} masks)")
            ax.set_xlabel("Spill Pixel Ratio"); ax.set_ylabel("Count"); ax.legend()
        else:
            ax.text(0.5, 0.5, "No masks found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            ax.set_title(label)
    plt.tight_layout()
    fig6.savefig(os.path.join(OUTPUT_DIR, "6_mask_coverage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig6); print("  Saved: 6_mask_coverage.png")

    # Figure 7 — Summary bar
    shift_metrics = {
        "Pixel Mean\nWasserstein":  wasserstein_distance([s["mean"] for s in stats1], [s["mean"] for s in stats2]),
        "Pixel Std\nWasserstein":   wasserstein_distance([s["std"]  for s in stats1], [s["std"]  for s in stats2]),
        "Contrast\nWasserstein":    tex_summary["contrast"]["wasserstein"],
        "Homogeneity\nWasserstein": tex_summary["homogeneity"]["wasserstein"],
        "Energy\nWasserstein":      tex_summary["energy"]["wasserstein"],
        "FFT Low\nWasserstein":     fft_summary["fft_low_ratio"]["wasserstein"],
        "FFT High\nWasserstein":    fft_summary["fft_high_ratio"]["wasserstein"],
        "Hist R\nWasserstein":      hist_wasserstein[0],
        "Hist G\nWasserstein":      hist_wasserstein[1],
        "Hist B\nWasserstein":      hist_wasserstein[2],
    }
    fig7, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(list(shift_metrics.keys()), list(shift_metrics.values()),
                  color=["#e53935" if v > 0.05 else "#43a047" for v in shift_metrics.values()],
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="Threshold 0.05")
    ax.set_title("Domain Shift Summary — Wasserstein Distances per Feature\n"
                 "(Red = significant shift  |  Green = mild shift)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_xticks(range(len(shift_metrics)))
    ax.set_xticklabels(list(shift_metrics.keys()), fontsize=9)
    ax.legend()
    for bar, val in zip(bars, shift_metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig7.savefig(os.path.join(OUTPUT_DIR, "7_domain_shift_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig7); print("  Saved: 7_domain_shift_summary.png")

    # ═════════════════════════════════════════════════════════════════════════
    # REPORT
    # ═════════════════════════════════════════════════════════════════════════
    report_lines = []
    rl = report_lines.append

    rl("=" * 70)
    rl("  DOMAIN SHIFT ANALYSIS REPORT")
    rl("  Dataset 3: Sentinel-1 SAR — Refined Deep-SAR SOS Dataset (Zenodo 15298010)")
    rl("  Dataset 4: Sentinel-1 SAR — CSIRO Oil Spill Detection Dataset (Kaggle/CSIRO)")
    rl("=" * 70)

    rl("\n── File Counts ──────────────────────────────────────────────────────")
    rl(f"  D3 images : {len(imgs1):6d}    D3 masks : {len(masks1)}")
    rl(f"  D4 images : {len(imgs2):6d}    D4 masks : {len(masks2)}")

    rl("\n── Image Dimensions ─────────────────────────────────────────────────")
    for k in ["height", "width"]:
        s1 = stat_summary[k]["D3"]; s2 = stat_summary[k]["D4"]
        if s1 and s2:
            rl(f"  {k:12s}  D3: {s1['mean']:7.1f} ± {s1['std']:6.1f}    "
               f"D4: {s2['mean']:7.1f} ± {s2['std']:6.1f}")

    rl("\n── Channel Analysis ─────────────────────────────────────────────────")
    ch1 = [s["channels"] for s in stats1 if "channels" in s]
    ch2 = [s["channels"] for s in stats2 if "channels" in s]
    u1, c1 = np.unique(ch1, return_counts=True)
    u2, c2 = np.unique(ch2, return_counts=True)
    rl(f"  D3 channel types: { {int(u): int(c) for u, c in zip(u1, c1)} }")
    rl(f"  D4 channel types: { {int(u): int(c) for u, c in zip(u2, c2)} }")
    if len(u1) > 1:
        rl("  ⚠ WARNING: D3 has mixed channel types — force convert to single channel in dataloader")
    if len(u2) > 1:
        rl("  ⚠ WARNING: D4 has mixed channel types — force convert to single channel in dataloader")

    rl("\n── Pixel Statistics ─────────────────────────────────────────────────")
    for k in ["mean", "std", "dynamic_range", "mean_r", "mean_g", "mean_b"]:
        s1 = stat_summary[k]["D3"]; s2 = stat_summary[k]["D4"]
        if s1 and s2:
            v1 = [s[k] for s in stats1 if k in s]
            v2 = [s[k] for s in stats2 if k in s]
            wd = wasserstein_distance(v1, v2) if v1 and v2 else 0
            rl(f"  {k:18s}  D3: {s1['mean']:7.2f} ± {s1['std']:6.2f}    "
               f"D4: {s2['mean']:7.2f} ± {s2['std']:6.2f}    "
               f"Wasserstein: {wd:.4f}")

    rl("\n── GLCM Texture Features ────────────────────────────────────────────")
    rl(f"  {'Feature':15s}  {'D3 Mean':>10s}  {'D4 Mean':>10s}  "
       f"{'Wasserstein':>14s}  {'KS stat':>8s}  {'p-value':>8s}")
    for k in tex_keys:
        t = tex_summary[k]
        rl(f"  {k:15s}  {t['D3_mean']:10.4f}  {t['D4_mean']:10.4f}  "
           f"{t['wasserstein']:14.4f}  {t['ks_stat']:8.4f}  {t['ks_p']:8.2e}")

    rl("\n── FFT Frequency Analysis ───────────────────────────────────────────")
    for k, label in [("fft_low_ratio", "Low-freq ratio"), ("fft_high_ratio", "High-freq ratio")]:
        f = fft_summary[k]
        rl(f"  {label:20s}  D3: {f['D3_mean']:.4f}   D4: {f['D4_mean']:.4f}   "
           f"Wasserstein: {f['wasserstein']:.4f}")

    rl("\n── Histogram (Channel) Wasserstein Distances ────────────────────────")
    for c, cname in enumerate(["Red/B0", "Green/B1", "Blue/B2"]):
        rl(f"  Channel {cname:10s}  Wasserstein: {hist_wasserstein[c]:.6f}")

    rl("\n── MMD (Maximum Mean Discrepancy) ───────────────────────────────────")
    rl(f"  MMD² (RBF kernel, PCA-50 features) = {mmd_val:.6f}")
    if   mmd_val > 0.5:  rl("  → VERY HIGH domain shift (cross-modal, highly incompatible domains)")
    elif mmd_val > 0.1:  rl("  → HIGH domain shift (significant adaptation required)")
    elif mmd_val > 0.02: rl("  → MODERATE domain shift")
    else:                rl("  → LOW domain shift (domains are relatively similar)")

    rl("\n── Mask / Annotation Coverage ───────────────────────────────────────")
    for vals, label in [(cov1, "D3"), (cov2, "D4")]:
        if vals:
            rl(f"  {label}: mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  "
               f"std={np.std(vals):.4f}  max={np.max(vals):.4f}")
        else:
            rl(f"  {label}: No masks found.")

    rl("\n── Domain Shift Interpretation ──────────────────────────────────────")
    rl("""
  D3 (Zenodo 15298010) — Refined Deep-SAR SOS Dataset:
    • Sentinel-1 SAR imagery, single band grayscale
    • Binary pixel masks — oil spill vs ocean (manually corrected)
    • Used for YOLO detection + UNet segmentation training
    • Masks refined: 38% train + 50% val corrected from original

  D4 (CSIRO via Kaggle) — Sentinel-1 SAR Oil Spill Dataset:
    • Sentinel-1 SAR imagery, JPEG grayscale, 400x400px
    • Image-level binary labels only (oil / no-oil) — no pixel masks
    • Used for CNN Verifier (Stage 2) training only
    • 5630 images — Oil: 1905, No-Oil: 3725 (imbalanced)
    • Coverage: Australian waters, May 2015 - Aug 2022

  Domain Shift Type: SAME-MODAL + CROSS-ENVIRONMENT
    ┌──────────────────────┬───────────────────────┬─────────────────────────┐
    │ Property             │ D3 (Zenodo 15298010)  │ D4 (CSIRO/Kaggle)       │
    ├──────────────────────┼───────────────────────┼─────────────────────────┤
    │ Sensor modality      │ Sentinel-1 SAR        │ Sentinel-1 SAR          │
    │ Channels             │ mixed 1+3 (normalize) │ 3 (grayscale as RGB)    │
    │ Label type           │ Pixel masks           │ Image-level labels      │
    │ Environment          │ Open Ocean (global)   │ Australian waters       │
    │ Usage in pipeline    │ YOLO + UNet           │ CNN Verifier only       │
    │ Acquisition          │ All-weather radar     │ All-weather radar       │
    └──────────────────────┴───────────────────────┴─────────────────────────┘

  Preprocessing Required Before Training:
    1. Force all images to single channel grayscale in dataloader
    2. Normalize pixel values to 0-1 range (both datasets)
    3. Resize D4 from 400x400 to 256x256 to match D3
    4. Apply light Lee speckle filter to D3 to reduce high-freq noise gap
    5. Use weighted loss for D4 CNN training — 66% no-oil class imbalance""")
    rl("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(OUTPUT_DIR, "domain_shift_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved to: {report_path}")

    metrics = {
        "DATASET3_PATH": DATASET3_PATH,
        "DATASET4_PATH": DATASET4_PATH,
        "file_counts": {
            "D3_images": len(imgs1), "D3_masks": len(masks1),
            "D4_images": len(imgs2), "D4_masks": len(masks2),
        },
        "channel_analysis": {
            "D3": {int(u): int(c) for u, c in zip(u1, c1)},
            "D4": {int(u): int(c) for u, c in zip(u2, c2)},
        },
        "mmd_squared": float(mmd_val),
        "texture":     tex_summary,
        "fft":         fft_summary,
        "histogram_wasserstein": {
            "channel_R": hist_wasserstein[0],
            "channel_G": hist_wasserstein[1],
            "channel_B": hist_wasserstein[2],
        },
        "mask_coverage": {
            "D3_mean": float(np.mean(cov1)) if cov1 else None,
            "D4_mean": float(np.mean(cov2)) if cov2 else None,
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "domain_shift_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved to: {json_path}")
    print(f"\n  All outputs saved in: {OUTPUT_DIR}")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()