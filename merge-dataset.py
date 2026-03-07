#!/usr/bin/env python3
"""
Merge Dataset Script

This script merges the Zenodo dataset (dataset_1) and Kaggle dataset (dataset_2)
into a unified merged-dataset directory with proper train/val/test splits and masks.

The script handles:
- Merging images from both datasets
- Generating masks for images without existing masks using thresholding
- Maintaining proper directory structure
- Creating train/val/test splits (70/15/15)
"""

import os
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import argparse

class DatasetMerger:
    def __init__(self, zenodo_path: str = "dataset_1", kaggle_path: str = "dataset_2", 
                 output_path: str = "merged-dataset", test_split: float = 0.15, 
                 val_split: float = 0.15, seed: int = 42):
        """
        Initialize the dataset merger.
        
        Args:
            zenodo_path: Path to Zenodo dataset
            kaggle_path: Path to Kaggle dataset  
            output_path: Output directory for merged dataset
            test_split: Fraction for test split
            val_split: Fraction for validation split
            seed: Random seed for reproducibility
        """
        self.zenodo_path = Path(zenodo_path)
        self.kaggle_path = Path(kaggle_path)
        self.output_path = Path(output_path)
        self.test_split = test_split
        self.val_split = val_split
        self.train_split = 1.0 - test_split - val_split
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Statistics tracking
        self.stats = {
            'zenodo_images': 0,
            'kaggle_images': 0,
            'total_images': 0,
            'images_with_masks': 0,
            'images_generated_masks': 0,
            'mask_generation_failures': 0
        }
    
    def find_images_and_masks(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        """
        Find all images and their corresponding masks in a dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of (image_path, mask_path) tuples
        """
        image_mask_pairs = []
        
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Search for images recursively
        for img_path in dataset_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                # Look for corresponding mask
                mask_path = self.find_corresponding_mask(img_path, dataset_path)
                image_mask_pairs.append((img_path, mask_path))
                
        return image_mask_pairs
    
    def find_corresponding_mask(self, image_path: Path, dataset_path: Path) -> Path:
        """
        Find the corresponding mask for an image.
        
        Args:
            image_path: Path to image file
            dataset_path: Root dataset path
            
        Returns:
            Path to mask file or None if not found
        """
        # Common mask naming patterns
        mask_patterns = [
            # Same name with _mask suffix
            image_path.stem + '_mask' + image_path.suffix,
            # Same name with mask prefix
            'mask_' + image_path.name,
            # Same name in masks subdirectory
            'masks/' + image_path.name,
            # Same name in annotations subdirectory  
            'annotations/' + image_path.name,
            # Same name with .png extension (common for masks)
            image_path.stem + '.png'
        ]
        
        # Search in current directory and common subdirectories
        search_paths = [
            image_path.parent,
            dataset_path / 'masks',
            dataset_path / 'annotations', 
            dataset_path / 'labels',
            dataset_path / 'ground_truth'
        ]
        
        for search_path in search_paths:
            for pattern in mask_patterns:
                mask_path = search_path / pattern
                if mask_path.exists():
                    return mask_path
        
        return None
    
    def generate_mask_from_image(self, image_path: Path) -> np.ndarray:
        """
        Generate a mask for an oil spill image using thresholding techniques.
        
        This uses multiple approaches to detect dark oil spill regions:
        1. Grayscale thresholding for dark regions
        2. Morphological operations to clean up noise
        3. Connected component analysis to find large regions
        
        Args:
            image_path: Path to input image
            
        Returns:
            Binary mask as numpy array
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Otsu's thresholding for automatic threshold selection
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Method 2: Adaptive thresholding for varying illumination
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
            
            # Combine both methods
            combined = cv2.bitwise_or(thresh_otsu, adaptive_thresh)
            
            # Morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            
            # Remove small noise
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            # Fill small holes
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
            
            # Create output mask
            mask = np.zeros_like(gray)
            
            # Filter components based on area and shape
            min_area = 100  # Minimum area for oil spill regions
            max_area = gray.shape[0] * gray.shape[1] * 0.8  # Maximum 80% of image
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                
                if min_area < area < max_area:
                    # Check if the region is sufficiently dark
                    mask_component = (labels == i).astype(np.uint8) * 255
                    
                    # Calculate mean intensity of the region in original image
                    region_mean = cv2.mean(gray, mask=mask_component)[0]
                    
                    # Oil spills are typically dark regions
                    if region_mean < 100:  # Threshold for dark regions
                        mask = cv2.bitwise_or(mask, mask_component)
            
            # Final cleanup
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            print(f"Error generating mask for {image_path}: {e}")
            return None
    
    def create_directory_structure(self):
        """Create the output directory structure."""
        splits = ['train', 'val', 'test']
        subdirs = ['images', 'masks']
        
        for split in splits:
            for subdir in subdirs:
                (self.output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def split_dataset(self, image_mask_pairs: List[Tuple[Path, Path]]) -> dict:
        """
        Split dataset into train/val/test sets.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            
        Returns:
            Dictionary with split assignments
        """
        # Shuffle the dataset
        shuffled_pairs = image_mask_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        total = len(shuffled_pairs)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)
        
        splits = {
            'train': shuffled_pairs[:train_end],
            'val': shuffled_pairs[train_end:val_end],
            'test': shuffled_pairs[val_end:]
        }
        
        return splits
    
    def process_and_copy_files(self, splits: dict):
        """
        Process and copy files to the output directory.
        
        Args:
            splits: Dictionary with split assignments
        """
        counter = 1
        
        for split_name, pairs in splits.items():
            print(f"\nProcessing {split_name} split ({len(pairs)} images)...")
            
            for img_path, mask_path in pairs:
                try:
                    # Generate new filename
                    new_filename = f"image_{counter:06d}.jpg"
                    
                    # Copy image
                    src_img = img_path
                    dst_img = self.output_path / split_name / 'images' / new_filename
                    shutil.copy2(src_img, dst_img)
                    
                    # Handle mask
                    if mask_path and mask_path.exists():
                        # Copy existing mask
                        dst_mask = self.output_path / split_name / 'masks' / new_filename
                        shutil.copy2(mask_path, dst_mask)
                        self.stats['images_with_masks'] += 1
                    else:
                        # Generate mask
                        generated_mask = self.generate_mask_from_image(src_img)
                        
                        if generated_mask is not None:
                            dst_mask = self.output_path / split_name / 'masks' / new_filename
                            cv2.imwrite(str(dst_mask), generated_mask)
                            self.stats['images_generated_masks'] += 1
                        else:
                            self.stats['mask_generation_failures'] += 1
                            print(f"Warning: Could not generate mask for {img_path}")
                    
                    counter += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    def merge_datasets(self):
        """Main method to merge datasets."""
        print("Starting dataset merger...")
        print(f"Zenodo dataset: {self.zenodo_path}")
        print(f"Kaggle dataset: {self.kaggle_path}")
        print(f"Output directory: {self.output_path}")
        
        # Check if input directories exist
        if not self.zenodo_path.exists():
            raise FileNotFoundError(f"Zenodo dataset directory not found: {self.zenodo_path}")
        
        if not self.kaggle_path.exists():
            raise FileNotFoundError(f"Kaggle dataset directory not found: {self.kaggle_path}")
        
        # Create output directory structure
        self.create_directory_structure()
        
        # Find images and masks in both datasets
        print("\nScanning Zenodo dataset...")
        zenodo_pairs = self.find_images_and_masks(self.zenodo_path)
        self.stats['zenodo_images'] = len(zenodo_pairs)
        print(f"Found {len(zenodo_pairs)} images in Zenodo dataset")
        
        print("\nScanning Kaggle dataset...")
        kaggle_pairs = self.find_images_and_masks(self.kaggle_path)
        self.stats['kaggle_images'] = len(kaggle_pairs)
        print(f"Found {len(kaggle_pairs)} images in Kaggle dataset")
        
        # Combine datasets
        all_pairs = zenodo_pairs + kaggle_pairs
        self.stats['total_images'] = len(all_pairs)
        
        print(f"\nTotal images found: {len(all_pairs)}")
        print(f"Zenodo: {len(zenodo_pairs)}, Kaggle: {len(kaggle_pairs)}")
        
        if len(all_pairs) == 0:
            print("No images found in either dataset!")
            return
        
        # Split dataset
        splits = self.split_dataset(all_pairs)
        
        # Process and copy files
        self.process_and_copy_files(splits)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("DATASET MERGER STATISTICS")
        print("="*50)
        print(f"Zenodo images processed: {self.stats['zenodo_images']}")
        print(f"Kaggle images processed: {self.stats['kaggle_images']}")
        print(f"Total images: {self.stats['total_images']}")
        print(f"Images with existing masks: {self.stats['images_with_masks']}")
        print(f"Images with generated masks: {self.stats['images_generated_masks']}")
        print(f"Mask generation failures: {self.stats['mask_generation_failures']}")
        
        # Calculate split sizes
        total = self.stats['total_images']
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        test_size = total - train_size - val_size
        
        print(f"\nSplit distribution:")
        print(f"Train: {train_size} images ({self.train_split*100:.1f}%)")
        print(f"Val:   {val_size} images ({self.val_split*100:.1f}%)")
        print(f"Test:  {test_size} images ({(1-self.train_split-self.val_split)*100:.1f}%)")
        
        print(f"\nOutput directory: {self.output_path}")
        print("Dataset merger completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Merge Zenodo and Kaggle oil spill datasets')
    parser.add_argument('--zenodo-path', type=str, default='dataset_1', 
                       help='Path to Zenodo dataset (default: dataset_1)')
    parser.add_argument('--kaggle-path', type=str, default='dataset_2',
                       help='Path to Kaggle dataset (default: dataset_2)')
    parser.add_argument('--output-path', type=str, default='merged-dataset',
                       help='Output directory for merged dataset (default: merged-dataset)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split fraction (default: 0.15)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split fraction (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate splits
    if args.test_split + args.val_split >= 1.0:
        raise ValueError("Test and validation splits must sum to less than 1.0")
    
    # Create merger and run
    merger = DatasetMerger(
        zenodo_path=args.zenodo_path,
        kaggle_path=args.kaggle_path,
        output_path=args.output_path,
        test_split=args.test_split,
        val_split=args.val_split,
        seed=args.seed
    )
    
    merger.merge_datasets()


if __name__ == "__main__":
    main()