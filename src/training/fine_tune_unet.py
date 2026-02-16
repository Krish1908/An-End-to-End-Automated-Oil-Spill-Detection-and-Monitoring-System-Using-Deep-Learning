# /content/drive/MyDrive/OIL-SPILL-8/src/training/fine_tune_unet.py

import os
import sys
import tensorflow as tf

# ---------------------------------------------------
# FINE-TUNING U-NET TRAINING SCRIPT
# ---------------------------------------------------

# Add project root to path
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
sys.path.append(PROJECT_ROOT)

from models.unet import build_unet, bce_dice_loss, dice_coef
from data.dataloader import create_unet_dataset

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-4
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (256, 256)

# Model paths
MODEL_DIR = "/content/drive/MyDrive/OIL-SPILL-8/models/unet"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fine-tuning paths
PROCESSED_DIR = "/content/drive/MyDrive/OIL-SPILL-8/src/data/processed"

# ---------------------------------------------------
# FINE-TUNING TRAINING
# ---------------------------------------------------
def main():
    """
    Main fine-tuning function for U-Net model
    """
    print("🎯 Starting U-Net Fine-Tuning")
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
    
    print(f"✅ Found {total_manual_images} manual images for fine-tuning across all directories")
    
    # Start fine-tuning training
    model = train_with_fine_tuning_main()
    
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
# Fine-Tuning Results

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
- Final model: {MODEL_DIR}/unet_fine_tuned.keras

## Training Summary
- Total epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}
- Base learning rate: {BASE_LEARNING_RATE}
- Fine-tuning: Enabled
"""
    
    with open(f"{MODEL_DIR}/fine_tuning_results.md", "w") as f:
        f.write(comparison_report)
    
    print(f"\n📄 Detailed results saved to: {MODEL_DIR}/fine_tuning_results.md")
    print("\n✅ Fine-tuning training completed successfully!")

# ---------------------------------------------------
# QUICK FINE-TUNING FUNCTION (for immediate use)
# ---------------------------------------------------
def quick_fine_tune_unet():
    """
    Quick function to fine-tune existing U-Net model with manual images
    """
    print("⚡ Quick Fine-Tuning for U-Net")
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
    
    # Create fine-tuned dataset using the manual images
    print("🔄 Creating dataset with manual images...")
    
    train_ds = create_unet_dataset("train", batch_size=BATCH_SIZE, augment=True)
    val_ds = create_unet_dataset("val", batch_size=BATCH_SIZE, augment=False)
    
    # Fine-tuning configuration
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{MODEL_DIR}/unet_quick_fine_tuned.keras",
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
    
    # Train with fewer epochs for quick fine-tuning
    print("🚀 Starting quick fine-tuning training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # Quick fine-tuning
        callbacks=callbacks
    )
    
    # Evaluate on test-new
    print("\n📊 Evaluating on test-new...")
    test_new_ds = create_unet_dataset("test-new", batch_size=BATCH_SIZE, augment=False)
    test_results = model.evaluate(test_new_ds, verbose=1)
    print(f"Test-new Dice: {test_results[1]:.4f}")
    
    print(f"\n✅ Quick fine-tuning completed!")
    print(f"Model saved to: {MODEL_DIR}/unet_quick_fine_tuned.keras")
    
    return model

# ---------------------------------------------------
# FINE-TUNING TRAINING MAIN FUNCTION
# ---------------------------------------------------
def train_with_fine_tuning_main():
    """
    Complete fine-tuning function for U-Net model
    """
    print("🚀 Starting Fine-Tuning Training...")
    
    # Import here to avoid circular imports
    import sys
    sys.path.append("/content/drive/MyDrive/OIL-SPILL-8/src")
    from models.unet import build_unet, bce_dice_loss, dice_coef
    
    # Configuration
    BATCH_SIZE = 8
    BASE_LEARNING_RATE = 1e-4
    EPOCHS = 20
    
    # Load existing model or build new one
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
    
    # Compile with fine-tuning learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    
    # Training phases with gradual fine-tuning
    phases = [
        {'name': 'phase1', 'epochs': 8, 'lr': BASE_LEARNING_RATE},
        {'name': 'phase2', 'epochs': 7, 'lr': BASE_LEARNING_RATE * 0.5},
        {'name': 'phase3', 'epochs': 5, 'lr': BASE_LEARNING_RATE * 0.25},
    ]
    
    current_epoch = 0
    best_model_path = "/content/drive/MyDrive/OIL-SPILL-8/models/unet/unet_fine_tuned.keras"
    
    for phase in phases:
        print(f"\n{'='*50}")
        print(f"🎯 PHASE: {phase['name']}")
        print(f"Epochs: {phase['epochs']}")
        print(f"Learning Rate: {phase['lr']}")
        print(f"{'='*50}")
        
        # Create training dataset
        train_ds = create_unet_dataset("train", batch_size=BATCH_SIZE, augment=True)
        
        # Validation dataset
        val_ds = create_unet_dataset("val", batch_size=BATCH_SIZE, augment=False)
        
        # Adjust learning rate
        model.optimizer.learning_rate.assign(phase['lr'])
        print(f"Learning rate: {phase['lr']:.6f}")
        
        # Callbacks for this phase
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{MODEL_DIR}/unet_fine_tune_{phase['name']}.keras",
                save_best_only=True,
                monitor="val_dice_coef",
                mode="max",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice_coef",
                factor=0.5,
                patience=2,
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
# MAIN EXECUTION
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="U-Net Fine-Tuning")
    parser.add_argument("--quick", action="store_true", help="Run quick fine-tuning (15 epochs)")
    parser.add_argument("--full", action="store_true", help="Run full fine-tuning (20 epochs)")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_fine_tune_unet()
    elif args.full or not any([args.quick, args.full]):
        main()
    else:
        print("Please specify --quick or --full")