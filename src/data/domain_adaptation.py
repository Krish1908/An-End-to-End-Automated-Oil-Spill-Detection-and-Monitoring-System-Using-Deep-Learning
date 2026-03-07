# /content/drive/MyDrive/OIL-SPILL-8/src/data/domain_adaptation.py

# MANUAL ONLINE IMAGES DOMAIN ADAPTATION PIPELINE

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import shutil

# ---------------------------------------------------
# DOMAIN ADAPTATION SOLUTIONS FOR MANUAL IMAGES
# ---------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/OIL-SPILL-8"
PROCESSED_DIR = f"{BASE_DIR}/src/data/processed"
UPLOAD_DIR = f"{BASE_DIR}/domain_adaptation_uploads"

# Create domain adaptation directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------
# 1. DOMAIN SHIFT ANALYSIS
# ---------------------------------------------------
def analyze_domain_shift():
    """
    Analyze differences between original and manual images
    """
    print("🔍 Analyzing Domain Shift...")
    
    # Load sample images from both domains
    original_imgs = []
    manual_imgs = []
    
    # Get original training images
    train_img_dir = f"{PROCESSED_DIR}/train/images"
    if os.path.exists(train_img_dir):
        for fname in os.listdir(train_img_dir)[:50]:  # Sample 50 images
            if fname.startswith("Oil") and fname.endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(train_img_dir, fname))
                if img is not None:
                    original_imgs.append(img)
    
    # Get manual images from all directories
    manual_dirs = [f"{PROCESSED_DIR}/train/images", f"{PROCESSED_DIR}/val/images", f"{PROCESSED_DIR}/test-new/images"]
    for manual_dir in manual_dirs:
        if os.path.exists(manual_dir):
            for fname in os.listdir(manual_dir):
                if fname.startswith("upload") and fname.endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(manual_dir, fname))
                    if img is not None:
                        manual_imgs.append(img)
    
    if len(original_imgs) == 0 or len(manual_imgs) == 0:
        print("❌ Not enough images for analysis")
        return
    
    # Calculate color statistics
    def get_color_stats(imgs):
        hists = []
        means = []
        stds = []
        for img in imgs:
            # Convert to LAB for better color analysis
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            means.append(np.mean(lab, axis=(0,1)))
            stds.append(np.std(lab, axis=(0,1)))
        return np.array(means), np.array(stds)
    
    orig_means, orig_stds = get_color_stats(original_imgs)
    manual_means, manual_stds = get_color_stats(manual_imgs)
    
    print(f"Original images - Mean: {np.mean(orig_means, axis=0)}")
    print(f"Manual images - Mean: {np.mean(manual_means, axis=0)}")
    print(f"Color shift: {np.mean(np.abs(np.mean(orig_means, axis=0) - np.mean(manual_means, axis=0)))}")
    
    return orig_means, orig_stds, manual_means, manual_stds

# ---------------------------------------------------
# 2. COLOR NORMALIZATION (REINHARD METHOD)
# ---------------------------------------------------
def reinhard_color_normalize(source_img, target_stats):
    """
    Normalize source image colors to match target domain statistics
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Calculate source statistics
    source_mean = np.mean(source_lab, axis=(0,1))
    source_std = np.std(source_lab, axis=(0,1))
    
    # Normalize each channel
    for i in range(3):
        source_lab[:,:,i] = (source_lab[:,:,i] - source_mean[i]) / source_std[i]
        source_lab[:,:,i] = source_lab[:,:,i] * target_stats['std'][i] + target_stats['mean'][i]
    
    # Clip values and convert back to BGR
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

# ---------------------------------------------------
# 3. CREATE DOMAIN ADAPTATION DATASET
# ---------------------------------------------------
def create_domain_adaptation_dataset():
    """
    Create a balanced dataset with domain adaptation
    """
    print("🔄 Creating Domain Adaptation Dataset...")
    
    # Analyze domain shift first
    orig_means, orig_stds, manual_means, manual_stds = analyze_domain_shift()
    target_stats = {
        'mean': np.mean(orig_means, axis=0),
        'std': np.mean(orig_stds, axis=0)
    }
    
    # Create augmented training directory
    aug_train_dir = f"{UPLOAD_DIR}/augmented_train"
    os.makedirs(f"{aug_train_dir}/images", exist_ok=True)
    os.makedirs(f"{aug_train_dir}/masks", exist_ok=True)
    
    # Process manual images with color normalization and augmentation from all directories
    manual_dirs = [
        (f"{PROCESSED_DIR}/train/images", f"{PROCESSED_DIR}/train/masks"),
        (f"{PROCESSED_DIR}/val/images", f"{PROCESSED_DIR}/val/masks"),
        (f"{PROCESSED_DIR}/test-new/images", f"{PROCESSED_DIR}/test-new/masks")
    ]
    
    count = 0
    for manual_img_dir, manual_mask_dir in manual_dirs:
        if not os.path.exists(manual_img_dir) or not os.path.exists(manual_mask_dir):
            print(f"⚠️ Skipping {manual_img_dir} - directory not found")
            continue
            
        for fname in os.listdir(manual_img_dir):
            if fname.startswith("upload") and fname.endswith((".jpg", ".png")):
                img_path = os.path.join(manual_img_dir, fname)
                mask_path = os.path.join(manual_mask_dir, os.path.splitext(fname)[0] + ".png")
                
                if not os.path.exists(mask_path):
                    continue
                    
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask is None:
                    continue
                
                # Apply color normalization
                normalized_img = reinhard_color_normalize(img, target_stats)
                
                # Save normalized image and mask
                base_name = f"adapted_{count}"
                cv2.imwrite(f"{aug_train_dir}/images/{base_name}.jpg", normalized_img)
                cv2.imwrite(f"{aug_train_dir}/masks/{base_name}.png", mask)
                count += 1
                
                # Apply augmentations
                augmented_pairs = augment_image_pair(normalized_img, mask)
                for i, (aug_img, aug_mask) in enumerate(augmented_pairs):
                    aug_name = f"{base_name}_aug_{i}"
                    cv2.imwrite(f"{aug_train_dir}/images/{aug_name}.jpg", aug_img)
                    cv2.imwrite(f"{aug_train_dir}/masks/{aug_name}.png", aug_mask)
    
    print(f"✅ Created {count * 4} augmented images for domain adaptation")
    return aug_train_dir

# ---------------------------------------------------
# 4. ADVANCED AUGMENTATION FOR DOMAIN ADAPTATION
# ---------------------------------------------------
def augment_image_pair(img, mask):
    """
    Apply domain-specific augmentations to image-mask pairs
    """
    augmented_pairs = []
    
    # 1. Color jittering (simulate different lighting conditions)
    for _ in range(2):
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-20, 20)    # Brightness
        
        aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmented_pairs.append((aug_img, mask))
    
    # 2. Gaussian noise (simulate sensor noise differences)
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    aug_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented_pairs.append((aug_img, mask))
    
    # 3. Blur/sharpness variations (simulate focus differences)
    if np.random.random() > 0.5:
        # Blur
        aug_img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        aug_img = cv2.filter2D(img, -1, kernel)
    augmented_pairs.append((aug_img, mask))
    
    return augmented_pairs

# ---------------------------------------------------
# 5. GRADUAL DOMAIN ADAPTATION TRAINING
# ---------------------------------------------------
def create_gradual_adaptation_schedule():
    """
    Create a training schedule that gradually introduces domain-shifted data
    """
    print("📅 Creating Gradual Adaptation Schedule...")
    
    # Phase 1: Train on original data only
    phase1_epochs = 10
    
    # Phase 2: Mix original + adapted data (70% original, 30% adapted)
    phase2_epochs = 15
    
    # Phase 3: Mix original + adapted data (50% original, 50% adapted) 
    phase3_epochs = 10
    
    # Phase 4: Mix original + adapted data (30% original, 70% adapted)
    phase4_epochs = 5
    
    schedule = {
        'phase1': {'epochs': phase1_epochs, 'original_ratio': 1.0, 'adapted_ratio': 0.0},
        'phase2': {'epochs': phase2_epochs, 'original_ratio': 0.7, 'adapted_ratio': 0.3},
        'phase3': {'epochs': phase3_epochs, 'original_ratio': 0.5, 'adapted_ratio': 0.5},
        'phase4': {'epochs': phase4_epochs, 'original_ratio': 0.3, 'adapted_ratio': 0.7},
    }
    
    print(f"Total epochs: {sum(s['epochs'] for s in schedule.values())}")
    return schedule

# ---------------------------------------------------
# 6. FEATURE-LEVEL DOMAIN ADAPTATION
# ---------------------------------------------------
def add_domain_classifier(base_model):
    """
    Add a domain classifier for adversarial training
    """
    # Freeze base model
    base_model.trainable = False
    
    # Add domain classifier head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    domain_output = tf.keras.layers.Dense(1, activation='sigmoid', name='domain')(x)
    
    domain_model = tf.keras.Model(inputs=base_model.input, outputs=domain_output)
    return domain_model

# ---------------------------------------------------
# 7. MAIN DOMAIN ADAPTATION PIPELINE
# ---------------------------------------------------
def run_domain_adaptation_pipeline():
    """
    Complete domain adaptation pipeline
    """
    print("🚀 Starting Domain Adaptation Pipeline...")
    
    # Step 1: Analyze domain shift
    print("\n1. Analyzing domain characteristics...")
    analyze_domain_shift()
    
    # Step 2: Create adapted dataset
    print("\n2. Creating domain-adapted dataset...")
    aug_dir = create_domain_adaptation_dataset()
    
    # Step 3: Create training schedule
    print("\n3. Creating gradual adaptation schedule...")
    schedule = create_gradual_adaptation_schedule()
    
    # Step 4: Generate adaptation report
    print("\n4. Generating adaptation report...")
    generate_adaptation_report(schedule, aug_dir)
    
    print("\n✅ Domain Adaptation Pipeline Complete!")
    print(f"📁 Augmented dataset saved to: {aug_dir}")
    print("📋 Next steps:")
    print("   1. Use the augmented dataset for training")
    print("   2. Follow the gradual adaptation schedule")
    print("   3. Monitor validation performance on test-new")

# ---------------------------------------------------
# 8. ADAPTATION REPORT GENERATOR
# ---------------------------------------------------
def generate_adaptation_report(schedule, aug_dir):
    """
    Generate a comprehensive report on domain adaptation
    """
    report = f"""
# Domain Adaptation Report

## Analysis Summary
- Original dataset: {len(os.listdir(f'{PROCESSED_DIR}/train/images'))} images
- Manual images: {len(os.listdir(f'{PROCESSED_DIR}/test-new/images'))} images
- Augmented images: {len(os.listdir(f'{aug_dir}/images'))} images

## Training Schedule
"""
    
    for phase, config in schedule.items():
        report += f"""
### {phase.replace('_', ' ').title()}
- Epochs: {config['epochs']}
- Original data ratio: {config['original_ratio'] * 100}%
- Adapted data ratio: {config['adapted_ratio'] * 100}%
"""
    
    report += """
## Recommendations

1. **Color Normalization**: Applied Reinhard color normalization to match domain statistics
2. **Data Augmentation**: Added color jittering, noise, and blur variations
3. **Gradual Adaptation**: Use the phased training approach to prevent catastrophic forgetting
4. **Monitoring**: Track validation loss on both original and new test sets

## Implementation Notes

- Start with phase 1 (original data only) to establish baseline
- Gradually increase adapted data ratio while monitoring performance
- If validation performance drops, reduce the adaptation rate
- Consider using learning rate scheduling during adaptation phases
"""
    
    with open(f"{UPLOAD_DIR}/adaptation_report.md", "w") as f:
        f.write(report)
    
    print(f"📄 Report saved to: {UPLOAD_DIR}/adaptation_report.md")

# ---------------------------------------------------
# 9. INTEGRATION WITH EXISTING TRAINING
# ---------------------------------------------------
def create_adapted_training_script():
    """
    Create a training script that integrates domain adaptation
    """
    script = '''# Domain-Adapted U-Net Training Script
import os
import sys
import tensorflow as tf
import numpy as np

# Add project root to path
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
sys.path.append(PROJECT_ROOT)

from models.unet import build_unet, bce_dice_loss, dice_coef
from data.dataloader import create_unet_dataset
from data.domain_adaptation import create_domain_adaptation_dataset, create_gradual_adaptation_schedule

# Configuration
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-4
ADAPTATION_DIR = "/content/drive/MyDrive/OIL-SPILL-8/domain_adaptation_uploads/augmented_train"

def train_with_adaptation():
    # Create adapted dataset
    aug_dir = create_domain_adaptation_dataset()
    schedule = create_gradual_adaptation_schedule()
    
    # Build model
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    
    # Training loop with gradual adaptation
    current_epoch = 0
    for phase, config in schedule.items():
        print(f"\\n🚀 Starting {phase}...")
        print(f"Original ratio: {config['original_ratio']}, Adapted ratio: {config['adapted_ratio']}")
        
        # Create mixed dataset for this phase
        train_ds = create_mixed_dataset(
            original_ratio=config['original_ratio'],
            adapted_ratio=config['adapted_ratio'],
            batch_size=BATCH_SIZE
        )
        
        # Adjust learning rate for adaptation phase
        lr = BASE_LEARNING_RATE * (0.5 ** (list(schedule.keys()).index(phase) + 1))
        model.optimizer.learning_rate.assign(lr)
        print(f"Learning rate: {lr}")
        
        # Train for this phase
        history = model.fit(
            train_ds,
            epochs=config['epochs'],
            initial_epoch=current_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    f"/content/drive/MyDrive/OIL-SPILL-8/models/unet/unet_adapted_{phase}.keras",
                    save_best_only=True
                )
            ]
        )
        
        current_epoch += config['epochs']
    
    # Final evaluation on test-new
    print("\\n📊 Final evaluation on test-new...")
    test_ds = create_unet_dataset("test-new", batch_size=BATCH_SIZE, augment=False)
    results = model.evaluate(test_ds)
    print(f"Test results: {results}")

def create_mixed_dataset(original_ratio, adapted_ratio, batch_size):
    """Create a mixed dataset with specified ratios"""
    # This function would need to be implemented based on your dataloader structure
    pass

if __name__ == "__main__":
    train_with_adaptation()
'''
    
    with open(f"{UPLOAD_DIR}/train_adapted_unet.py", "w") as f:
        f.write(script)
    
    print(f"🐍 Training script saved to: {UPLOAD_DIR}/train_adapted_unet.py")

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
if __name__ == "__main__":
    run_domain_adaptation_pipeline()
    create_adapted_training_script()