# /content/drive/MyDrive/OIL-SPILL-8/src/training/train_domain_adapted_unet.py

import os
import sys
import tensorflow as tf

# ---------------------------------------------------
# DOMAIN-ADAPTED U-NET TRAINING SCRIPT
# ---------------------------------------------------

# Add project root to path
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
sys.path.append(PROJECT_ROOT)

from models.unet import build_unet, bce_dice_loss, dice_coef
from data.dataloader_rgb import create_unet_dataset

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-4
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (256, 256)

# Model paths
MODEL_DIR = "/content/drive/MyDrive/OIL-SPILL-8/models/unet"
os.makedirs(MODEL_DIR, exist_ok=True)

# Domain adaptation paths (updated based on actual output)
ADAPTATION_DIR = "/content/drive/MyDrive/OIL-SPILL-8/domain_adaptation_uploads/augmented_train"
PROCESSED_DIR = "/content/drive/MyDrive/OIL-SPILL-8/src/data/processed"

# ---------------------------------------------------
# DOMAIN ADAPTATION TRAINING
# ---------------------------------------------------
def main():
    """
    Main training function with domain adaptation
    """
    print("🎯 Starting Domain-Adapted U-Net Training")
    print("=" * 60)
    
    # Check if manual images exist in all directories
    manual_dirs = [
        f"{PROCESSED_DIR}/train/images",
        f"{PROCESSED_DIR}/val/images", 
        f"{PROCESSED_DIR}/test-new/images"
    ]
    
    total_manual_images = 0
    for manual_dir in manual_dirs:
        if os.path.exists(manual_dir):
            upload_files = [f for f in os.listdir(manual_dir) if f.startswith("upload")]
            total_manual_images += len(upload_files)
    
    if total_manual_images == 0:
        print("❌ No manual images found in any directory")
        print("Please add your upload-xx images to: train/images, val/images, or test-new/images")
        return
    
    print(f"✅ Found {total_manual_images} manual images for adaptation across all directories")
    
    # Start domain adaptation training
    model = train_with_domain_adaptation_main()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    # Load test-new dataset using standard dataloader
    test_new_ds = create_unet_dataset("test-new", batch_size=BATCH_SIZE, augment=False)
    
    # Evaluate on test-new
    print("\n🔍 Evaluating on test-new (manual images)...")
    test_results = model.evaluate(test_new_ds, verbose=1)
    print(f"Test-new Loss: {test_results[0]:.4f}")
    print(f"Test-new Dice: {test_results[1]:.4f}")
    
    # Compare with original test set
    print("\n🔍 Evaluating on original test set...")
    test_ds = create_unet_dataset("test", batch_size=BATCH_SIZE, augment=False)
    original_test_results = model.evaluate(test_ds, verbose=1)
    print(f"Original Test Loss: {original_test_results[0]:.4f}")
    print(f"Original Test Dice: {original_test_results[1]:.4f}")
    
    # Save comparison results
    comparison_report = f"""
# Domain Adaptation Results

## Test Performance Comparison

| Dataset | Loss | Dice Coefficient |
|---------|------|------------------|
| Original Test | {original_test_results[0]:.4f} | {original_test_results[1]:.4f} |
| Test-New (Manual) | {test_results[0]:.4f} | {test_results[1]:.4f} |

## Improvement Analysis

- **Original Test Dice**: {original_test_results[1]:.4f}
- **Test-New Dice**: {test_results[1]:.4f}
- **Improvement**: {test_results[1] - original_test_results[1]:.4f}

## Model Saved To
- Final model: {MODEL_DIR}/unet_domain_adapted.keras

## Training Summary
- Total epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}
- Base learning rate: {BASE_LEARNING_RATE}
- Domain adaptation: Enabled
"""
    
    with open(f"{MODEL_DIR}/domain_adaptation_results.md", "w") as f:
        f.write(comparison_report)
    
    print(f"\n📄 Detailed results saved to: {MODEL_DIR}/domain_adaptation_results.md")
    print("\n✅ Domain adaptation training completed successfully!")

# ---------------------------------------------------
# QUICK ADAPTATION FUNCTION (for immediate use)
# ---------------------------------------------------
def quick_adapt_unet():
    """
    Quick function to adapt existing U-Net model with manual images
    """
    print("⚡ Quick Domain Adaptation for U-Net")
    print("-" * 40)
    
    # Load existing model
    existing_model_path = "/content/drive/MyDrive/OIL-SPILL-8/models/unet/unet_segmentation.keras"
    if os.path.exists(existing_model_path):
        print("📦 Loading existing U-Net model...")
        model = tf.keras.models.load_model(
            existing_model_path,
            custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss},
            compile=False
        )
    else:
        print("🏗️ Building new U-Net model...")
        model = build_unet()
    
    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    
    # Create domain-adapted dataset using the augmented training data
    print("🔄 Creating mixed dataset with domain adaptation...")
    print(f"Using augmented dataset from: {ADAPTATION_DIR}")
    
    train_ds = create_mixed_unet_dataset(
        original_ratio=0.6,  # 60% original, 40% adapted
        adapted_ratio=0.4,
        batch_size=BATCH_SIZE,
        augment=True
    )
    
    val_ds = create_unet_dataset("val", batch_size=BATCH_SIZE, augment=False)
    
    # Fine-tuning configuration
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{MODEL_DIR}/unet_quick_adapted.keras",
            save_best_only=True,
            monitor="val_dice_coef",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            factor=0.5,
            patience=3,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            patience=5,
            mode="max",
            restore_best_weights=True
        )
    ]
    
    # Train with fewer epochs for quick adaptation
    print("🚀 Starting quick adaptation training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # Quick adaptation
        callbacks=callbacks
    )
    
    # Evaluate on test-new
    print("\n📊 Evaluating on test-new...")
    test_new_ds = create_unet_dataset("test-new", batch_size=BATCH_SIZE, augment=False)
    test_results = model.evaluate(test_new_ds, verbose=1)
    print(f"Test-new Dice: {test_results[1]:.4f}")
    
    print(f"\n✅ Quick adaptation completed!")
    print(f"Model saved to: {MODEL_DIR}/unet_quick_adapted.keras")
    
    return model

# ---------------------------------------------------
# DOMAIN ADAPTATION TRAINING MAIN FUNCTION
# ---------------------------------------------------
def train_with_domain_adaptation_main():
    """
    Complete training function with domain adaptation using standard dataloader
    """
    print("🚀 Starting Domain-Adapted Training...")
    
    # Import here to avoid circular imports
    import sys
    sys.path.append("/content/drive/MyDrive/OIL-SPILL-8/src")
    from models.unet import build_unet, bce_dice_loss, dice_coef
    
    # Configuration
    BATCH_SIZE = 8
    BASE_LEARNING_RATE = 1e-4
    EPOCHS = 40
    
    # Create model
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    
    # Training phases with gradual adaptation
    phases = [
        {'name': 'phase1', 'epochs': 10, 'original_ratio': 1.0, 'adapted_ratio': 0.0},
        {'name': 'phase2', 'epochs': 12, 'original_ratio': 0.8, 'adapted_ratio': 0.2},
        {'name': 'phase3', 'epochs': 10, 'original_ratio': 0.6, 'adapted_ratio': 0.4},
        {'name': 'phase4', 'epochs': 8, 'original_ratio': 0.4, 'adapted_ratio': 0.6},
    ]
    
    current_epoch = 0
    best_model_path = "/content/drive/MyDrive/OIL-SPILL-8/models/unet/unet_domain_adapted.keras"
    
    for phase in phases:
        print(f"\n{'='*50}")
        print(f"🎯 PHASE: {phase['name']}")
        print(f"Epochs: {phase['epochs']}")
        print(f"Original: {phase['original_ratio']*100}%, Adapted: {phase['adapted_ratio']*100}%")
        print(f"{'='*50}")
        
        # Create mixed dataset for this phase
        train_ds = create_mixed_unet_dataset(
            original_ratio=phase['original_ratio'],
            adapted_ratio=phase['adapted_ratio'],
            batch_size=BATCH_SIZE,
            augment=True
        )
        
        # Validation dataset (original)
        val_ds = create_unet_dataset(
            "val", 
            batch_size=BATCH_SIZE,
            augment=False
        )
        
        # Adjust learning rate
        lr = BASE_LEARNING_RATE * (0.8 ** phases.index(phase))
        model.optimizer.learning_rate.assign(lr)
        print(f"Learning rate: {lr:.6f}")
        
        # Callbacks for this phase
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{MODEL_DIR}/unet_{phase['name']}.keras",
                save_best_only=True,
                monitor="val_dice_coef",
                mode="max",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice_coef",
                factor=0.5,
                patience=3,
                mode="max",
                verbose=1
            )
        ]
        
        # Train this phase
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=current_epoch + phase['epochs'],
            initial_epoch=current_epoch,
            callbacks=callbacks
        )
        
        current_epoch += phase['epochs']
        
        # Evaluate on test-new after each phase
        print(f"\n📊 Evaluating on test-new after {phase['name']}...")
        test_new_ds = create_unet_dataset("test-new", batch_size=BATCH_SIZE, augment=False)
        test_results = model.evaluate(test_new_ds, verbose=0)
        print(f"Test-new results: Loss={test_results[0]:.4f}, Dice={test_results[1]:.4f}")
    
    # Save final model
    model.save(best_model_path)
    print(f"\n✅ Final model saved to: {best_model_path}")
    
    return model

# ---------------------------------------------------
# MIXED DATASET CREATOR (using standard dataloader)
# ---------------------------------------------------
def create_mixed_unet_dataset(original_ratio=0.7, adapted_ratio=0.3, 
                             batch_size=8, augment=False, min_oil_ratio=0.01):
    """
    Create a mixed dataset combining original and adapted data using standard dataloader
    """
    print(f"🔄 Creating mixed dataset: {original_ratio*100}% original, {adapted_ratio*100}% adapted")
    
    # Load original dataset using standard dataloader
    original_ds = create_unet_dataset("train", batch_size, augment, min_oil_ratio)
    
    # Load adapted dataset from augmented_train directory
    adapted_ds = create_adapted_unet_dataset(ADAPTATION_DIR, batch_size, augment)
    
    # Calculate dataset sizes and mixing
    original_size = len(original_ds) if hasattr(original_ds, '__len__') else 100
    adapted_size = len(adapted_ds) if hasattr(adapted_ds, '__len__') else 50
    
    # Mix datasets based on ratios
    if original_ratio > 0 and adapted_ratio > 0:
        # Calculate sampling probabilities
        total_ratio = original_ratio + adapted_ratio
        orig_weight = original_ratio / total_ratio
        adapt_weight = adapted_ratio / total_ratio
        
        # Mix datasets
        mixed_ds = tf.data.Dataset.sample_from_datasets(
            [original_ds, adapted_ds],
            weights=[orig_weight, adapt_weight]
        )
    elif original_ratio > 0:
        mixed_ds = original_ds
    else:
        mixed_ds = adapted_ds
    
    return mixed_ds.prefetch(AUTOTUNE)

# ---------------------------------------------------
# ADAPTED UNET DATASET (using augmented data)
# ---------------------------------------------------
def create_adapted_unet_dataset(aug_dir, batch_size=8, augment=False):
    """
    Create dataset from domain-adapted images using standard dataloader functions
    """
    import tensorflow as tf
    
    img_dir = f"{aug_dir}/images"
    mask_dir = f"{aug_dir}/masks"
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print("❌ Adapted dataset not found")
        return None
    
    image_paths = []
    mask_paths = []
    
    for fname in os.listdir(img_dir):
        if fname.endswith((".jpg", ".png")):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
            
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
    
    image_paths = tf.constant(image_paths)
    mask_paths = tf.constant(mask_paths)
    
    def load_item(img_path, mask_path):
        img = load_image(img_path)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0
        return img, mask
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_unet, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# ---------------------------------------------------
# COMMON FUNCTIONS (from standard dataloader)
# ---------------------------------------------------
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_unet(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    
    return img, mask

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain-Adapted U-Net Training")
    parser.add_argument("--quick", action="store_true", help="Run quick adaptation (15 epochs)")
    parser.add_argument("--full", action="store_true", help="Run full adaptation (40 epochs)")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_adapt_unet()
    elif args.full or not any([args.quick, args.full]):
        main()
    else:
        print("Please specify --quick or --full")
