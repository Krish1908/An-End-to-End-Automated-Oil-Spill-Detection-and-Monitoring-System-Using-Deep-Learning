# /content/drive/MyDrive/OIL-SPILL-8/src/data/fix_manual_masks.py

# MANUAL ONLINE IMAGE MASK GENERATION

import os
import cv2
import numpy as np

# ---------------------------------------------------
# FIX MANUAL MASKS FOR DOMAIN ADAPTATION
# ---------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/OIL-SPILL-8"
PROCESSED_DIR = f"{BASE_DIR}/src/data/processed"

# Directories to check for upload-xx files
DIRECTORIES_TO_CHECK = ["train", "val", "test-new"]

# ---------------------------------------------------
# VERIFY AND FIX MASKS FOR DIRECTORY
# ---------------------------------------------------
def verify_and_fix_masks_for_directory(split_name):
    """
    Verify that manual masks exist and fix any issues for a specific directory
    """
    print(f"🔧 Verifying and fixing manual masks for {split_name}...")
    
    split_dir = f"{PROCESSED_DIR}/{split_name}"
    img_dir = f"{split_dir}/images"
    mask_dir = f"{split_dir}/masks"
    
    # Create directories if they don't exist
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Check if upload images exist
    upload_images = [f for f in os.listdir(img_dir) if f.startswith("upload")]
    upload_masks = [f for f in os.listdir(mask_dir) if f.startswith("upload")]
    
    print(f"Found {len(upload_images)} upload images in {split_name}")
    print(f"Found {len(upload_masks)} upload masks in {split_name}")
    
    # Check for missing masks
    missing_masks = []
    for img_file in upload_images:
        mask_file = os.path.splitext(img_file)[0] + ".png"
        if mask_file not in upload_masks:
            missing_masks.append(img_file)
    
    if missing_masks:
        print(f"⚠️ Missing masks for {split_name}: {missing_masks}")
        print(f"Please ensure masks are created for all upload images in {split_name}")
        return False
    
    # Verify mask quality
    print(f"\n🔍 Analyzing mask quality for {split_name}...")
    for mask_file in upload_masks:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"❌ Invalid mask: {mask_file}")
            continue
            
        # Check mask properties
        total_pixels = mask.shape[0] * mask.shape[1]
        oil_pixels = np.sum(mask > 0)
        oil_ratio = oil_pixels / total_pixels
        
        print(f"📊 {split_name}/{mask_file}: {oil_pixels}/{total_pixels} pixels ({oil_ratio:.2%} oil)")
        
        # Check for common issues
        if oil_ratio == 0:
            print(f"⚠️  Warning: {mask_file} has no oil pixels")
        elif oil_ratio > 0.8:
            print(f"⚠️  Warning: {mask_file} has very high oil coverage ({oil_ratio:.2%})")
    
    print(f"\n✅ Mask verification completed for {split_name}")
    return True

# ---------------------------------------------------
# ENHANCE MASK QUALITY FOR DIRECTORY
# ---------------------------------------------------
def enhance_mask_quality_for_directory(split_name):
    """
    Enhance the quality of manual masks for a specific directory
    """
    print(f"🎨 Enhancing mask quality for {split_name}...")
    
    split_dir = f"{PROCESSED_DIR}/{split_name}"
    mask_dir = f"{split_dir}/masks"
    enhanced_dir = f"{split_dir}/masks_enhanced"
    os.makedirs(enhanced_dir, exist_ok=True)
    
    for mask_file in os.listdir(mask_dir):
        if not mask_file.startswith("upload"):
            continue
            
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((3,3), np.uint8)
        
        # Remove small noise
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Fill small holes
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Save enhanced mask
        enhanced_path = os.path.join(enhanced_dir, mask_file)
        cv2.imwrite(enhanced_path, mask_clean)
        
        print(f"✅ Enhanced: {split_name}/{mask_file}")
    
    print(f"🎉 Mask enhancement completed for {split_name}!")

# ---------------------------------------------------
# CREATE MASK STATISTICS FOR DIRECTORY
# ---------------------------------------------------
def create_mask_statistics_for_directory(split_name):
    """
    Create statistics about the manual masks for a specific directory
    """
    print(f"📈 Creating mask statistics for {split_name}...")
    
    split_dir = f"{PROCESSED_DIR}/{split_name}"
    mask_dir = f"{split_dir}/masks"
    stats = {
        'total_masks': 0,
        'total_pixels': 0,
        'total_oil_pixels': 0,
        'min_oil_ratio': 1.0,
        'max_oil_ratio': 0.0,
        'avg_oil_ratio': 0.0
    }
    
    oil_ratios = []
    
    for mask_file in os.listdir(mask_dir):
        if not mask_file.startswith("upload"):
            continue
            
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        stats['total_masks'] += 1
        total_pixels = mask.shape[0] * mask.shape[1]
        oil_pixels = np.sum(mask > 0)
        oil_ratio = oil_pixels / total_pixels
        
        stats['total_pixels'] += total_pixels
        stats['total_oil_pixels'] += oil_pixels
        oil_ratios.append(oil_ratio)
        
        stats['min_oil_ratio'] = min(stats['min_oil_ratio'], oil_ratio)
        stats['max_oil_ratio'] = max(stats['max_oil_ratio'], oil_ratio)
    
    if stats['total_masks'] > 0:
        stats['avg_oil_ratio'] = np.mean(oil_ratios)
    
    # Generate report
    report = f"""
# Manual Mask Statistics for {split_name}

## Summary
- Total masks: {stats['total_masks']}
- Total pixels: {stats['total_pixels']:,}
- Total oil pixels: {stats['total_oil_pixels']:,}

## Oil Coverage Analysis
- Minimum oil ratio: {stats['min_oil_ratio']:.2%}
- Maximum oil ratio: {stats['max_oil_ratio']:.2%}
- Average oil ratio: {stats['avg_oil_ratio']:.2%}

## Quality Assessment
"""
    
    if stats['avg_oil_ratio'] < 0.05:
        report += "- ⚠️  Low average oil coverage - may need more aggressive annotation\n"
    elif stats['avg_oil_ratio'] > 0.5:
        report += "- ⚠️  High average oil coverage - check for over-annotation\n"
    else:
        report += "- ✅ Good oil coverage distribution\n"
    
    if stats['min_oil_ratio'] == 0:
        report += "- ⚠️  Some masks have no oil pixels - verify annotations\n"
    
    if stats['max_oil_ratio'] > 0.8:
        report += "- ⚠️  Some masks have very high oil coverage - check for errors\n"
    
    # Save report
    with open(f"{split_dir}/mask_statistics.md", "w") as f:
        f.write(report)
    
    print(report)
    print(f"📄 Statistics saved to: {split_dir}/mask_statistics.md")

# ---------------------------------------------------
# VALIDATE MASK-IMAGE PAIRS FOR DIRECTORY
# ---------------------------------------------------
def validate_mask_image_pairs_for_directory(split_name):
    """
    Validate that each image has a corresponding mask with matching dimensions for a specific directory
    """
    print(f"🔍 Validating mask-image pairs for {split_name}...")
    
    split_dir = f"{PROCESSED_DIR}/{split_name}"
    img_dir = f"{split_dir}/images"
    mask_dir = f"{split_dir}/masks"
    
    issues = []
    
    for img_file in os.listdir(img_dir):
        if not img_file.startswith("upload"):
            continue
            
        base_name = os.path.splitext(img_file)[0]
        mask_file = base_name + ".png"
        
        # Check if mask exists
        if not os.path.exists(os.path.join(mask_dir, mask_file)):
            issues.append(f"Missing mask for {img_file}")
            continue
        
        # Check dimensions match
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            issues.append(f"Could not read files for {img_file}")
            continue
        
        if img.shape[:2] != mask.shape:
            issues.append(f"Dimension mismatch for {img_file}: img {img.shape[:2]}, mask {mask.shape}")
    
    if issues:
        print(f"❌ Found issues in {split_name}:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✅ All mask-image pairs are valid in {split_name}!")
        return True

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
def main():
    """
    Main function to run all mask validation and enhancement steps for all directories
    """
    print("🚀 Starting Manual Mask Quality Assurance")
    print("=" * 50)
    
    all_success = True
    
    # Process each directory
    for split_name in DIRECTORIES_TO_CHECK:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {split_name.upper()} DIRECTORY")
        print(f"{'='*60}")
        
        # Step 1: Validate pairs
        if not validate_mask_image_pairs_for_directory(split_name):
            print(f"❌ Please fix the above issues in {split_name} before proceeding")
            all_success = False
            continue
        
        # Step 2: Verify and fix masks
        if not verify_and_fix_masks_for_directory(split_name):
            print(f"❌ Please create missing masks in {split_name} before proceeding")
            all_success = False
            continue
        
        # Step 3: Enhance quality
        enhance_mask_quality_for_directory(split_name)
        
        # Step 4: Create statistics
        create_mask_statistics_for_directory(split_name)
    
    print("\n" + "=" * 60)
    if all_success:
        print("✅ Manual mask quality assurance completed successfully!")
        print("📁 Enhanced masks saved to:")
        for split_name in DIRECTORIES_TO_CHECK:
            print(f"   - {split_name}/masks_enhanced/")
        print("📄 Statistics saved to:")
        for split_name in DIRECTORIES_TO_CHECK:
            print(f"   - {split_name}/mask_statistics.md")
    else:
        print("⚠️  Some directories had issues. Please review the output above.")
        print("   Fix the issues and re-run the script.")
    
    print("\n💡 Next steps:")
    print("   1. Review the statistics reports for each directory")
    print("   2. Check enhanced masks if quality issues were found")
    print("   3. Proceed with domain adaptation pipeline")

if __name__ == "__main__":
    main()
