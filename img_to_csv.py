#!/usr/bin/env python3
"""
Image to CSV Converter: A script for converting images to CSV format

This script reads image files from a directory and converts them to a CSV file
where each row represents an image and each column represents a pixel value.
Optional functionality to include subject/class labels in the CSV.
"""

import os
import cv2
import numpy as np
import sys
import argparse
import csv
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Optional, Tuple


def read_image(file_path: Union[str, Path]) -> np.ndarray:
    """
    Read a image file and return as a numpy array.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Image as a numpy array
    """
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error reading {file_path}")
    return img


def extract_label_from_path(file_path: Path, pattern: str = None) -> Optional[str]:
    """
    Extract a label from the file path based on a pattern or directory name.
    
    Args:
        file_path: Path to the image file
        pattern: Pattern to use for label extraction (default: use parent directory name)
    
    Returns:
        Extracted label as string or None if not applicable
    """
    if pattern:
        # Extract label based on the specified pattern
        # This can be extended with regex or other pattern matching logic
        return None  # Placeholder for pattern-based extraction
    else:
        # Use parent directory name as label
        return file_path.parent.name


def convert_img_to_csv(input_dir: Union[str, Path], 
                       output_file: Union[str, Path],
                       file_extensions: List[str] = ['.pgm'],
                       recursive: bool = True,
                       include_labels: bool = True,
                       label_pattern: str = None,
                       include_filenames: bool = False,
                       resize: Optional[Tuple[int, int]] = None) -> int:
    """
    Convert all images in a directory to a single CSV file.
    
    Args:
        input_dir: Directory containing input images
        output_file: Path to save the CSV file
        file_extensions: List of file extensions to process
        recursive: Whether to process subdirectories
        include_labels: Whether to include labels in the CSV
        label_pattern: Pattern to use for label extraction
        include_filenames: Whether to include filenames in the CSV
        resize: Optional tuple (width, height) to resize images before flattening
        
    Returns:
        Number of images processed
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    # Find all matching image files
    all_files = []
    for ext in file_extensions:
        if recursive:
            all_files.extend(list(input_dir.glob(f"**/*{ext}")))
        else:
            all_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not all_files:
        print(f"No files with extensions {file_extensions} found in {input_dir}")
        return 0
    
    all_files = sorted(all_files)
    
    # Ensure output directory exists
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Process a single image to determine the header shape
    sample_img = read_image(all_files[0])
    if resize:
        sample_img = cv2.resize(sample_img, resize)
    num_pixels = sample_img.size
    
    # Set up the CSV file
    processed_count = 0
    
    with open(output_file, 'w', newline='') as csvfile:
        # Create header row
        header = []
        if include_filenames:
            header.append('filename')
        if include_labels:
            header.append('label')
        header.extend([i for i in range(num_pixels)])
        
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        
        # Process each image and write directly to CSV
        for file_path in tqdm(all_files, desc="Processing images", unit="file"):
            try:
                # Read the image
                img = read_image(file_path)
                
                # Resize if specified
                if resize:
                    img = cv2.resize(img, resize)
                
                # Flatten the image into a 1D array
                flattened_img = img.flatten()
                
                # Create a row with optional filename and label
                row = []
                
                if include_filenames:
                    row.append(str(file_path.relative_to(input_dir)))
                
                if include_labels:
                    label = extract_label_from_path(file_path, label_pattern)
                    if label:
                        row.append(label)
                    else:
                        row.append("")  # Empty label if not found
                
                # Add pixel values
                row.extend([int(pixel_val) for pixel_val in flattened_img])
                
                # Write row directly to CSV
                csv_writer.writerow(row)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Print dataset statistics
    if processed_count > 0:
        print(f"Saved {processed_count} images to {output_file}")
        print(f"Dataset dimensions: {processed_count} rows × {len(header)} columns")
    else:
        print("No data processed. CSV not created.")
    
    return processed_count


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert images to CSV format."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input directory containing images")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output CSV file path")
    parser.add_argument("--extensions", nargs="+", default=['.pgm'],
                        help="File extensions to process (default: .pgm)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't process subdirectories")
    parser.add_argument("--no-labels", action="store_true",
                        help="Don't include class/subject labels in CSV")
    parser.add_argument("--include-filenames", action="store_true",
                        help="Include filenames in the CSV")
    parser.add_argument("--resize", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help="Resize images to WIDTH x HEIGHT before processing")
    
    args = parser.parse_args()
    
    # Convert images to CSV
    num_processed = convert_img_to_csv(
        input_dir=args.input,
        output_file=args.output,
        file_extensions=args.extensions,
        recursive=not args.no_recursive,
        include_labels=not args.no_labels,
        include_filenames=args.include_filenames,
        resize=tuple(args.resize) if args.resize else None
    )
    
    if num_processed > 0:
        print(f"Conversion complete! {num_processed} images converted to CSV.")
    else:
        print("No images were converted.")


if __name__ == "__main__":
    # sys.argv = [sys.argv[0], 
    #             "-i", "cropped_downsampled", 
    #             "-o", "csvs/cropped_downsampled_full.csv", 
    #             "--extensions", ".pgm",
    #             "--no-labels", "--include-filenames"]

    # sys.argv = [sys.argv[0], 
    #             "-i", "cropped_downsampled", 
    #             "-o", "csvs/cropped_downsampled_A+000E+00.csv", 
    #             "--extensions", "A+000E+00.pgm",
    #             "--no-labels", "--include-filenames"]

    sys.argv = [sys.argv[0], 
                "-i", "cropped_downsampled", 
                "-o", "csvs/cropped_downsampled_P00.csv", 
                "--extensions", "P00*.pgm",
                "--no-labels", "--include-filenames"]

    main()
