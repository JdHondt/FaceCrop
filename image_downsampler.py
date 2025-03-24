#!/usr/bin/env python3
"""
ImageDownsampler: A utility for downsizing images to a target resolution

This module provides tools for batch resizing images with various options
for interpolation methods and handling of different image formats.
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Union, Optional


class ImageDownsampler:
    """A class for downsampling images to a target size."""
    
    def __init__(self, target_size: Tuple[int, int] = (100, 100),
                 interpolation: int = cv2.INTER_AREA,
                 preserve_aspect_ratio: bool = False,
                 grayscale: bool = False):
        """
        Initialize the ImageDownsampler with configurable parameters.
        
        Args:
            target_size: Output image size as (width, height)
            interpolation: OpenCV interpolation method
            preserve_aspect_ratio: Whether to maintain aspect ratio while resizing
            grayscale: Whether to convert images to grayscale
        """
        self.target_size = target_size
        self.interpolation = interpolation
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.grayscale = grayscale
        
        # Mapping of interpolation method names to OpenCV constants
        self.interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
    
    def downsample(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the target size.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image
        """
        # Convert to grayscale if requested
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.preserve_aspect_ratio:
            # Get current dimensions
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate new dimensions while preserving aspect ratio
            aspect = w / h
            if w / target_w > h / target_h:
                # Width is the constraining dimension
                new_w = target_w
                new_h = int(new_w / aspect)
            else:
                # Height is the constraining dimension
                new_h = target_h
                new_w = int(new_h * aspect)
            
            # Resize the image
            resized = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)
            
            # Create a black canvas of target size
            if len(image.shape) == 3:
                result = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
            else:
                result = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Calculate position to paste resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place the resized image on the canvas
            if len(image.shape) == 3:
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
            else:
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return result
        else:
            # Simple resize to target dimensions
            return cv2.resize(image, self.target_size, interpolation=self.interpolation)
    
    def process_image(self, image_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      return_image: bool = False) -> Optional[np.ndarray]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the resized image (if None, image is not saved)
            return_image: Whether to return the processed image
            
        Returns:
            If return_image is True, returns the processed image
            Otherwise returns None
        """
        try:
            # Read image
            if self.grayscale:
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(image_path))
                
            if img is None:
                raise ValueError(f"Error reading {image_path}")
            
            # Resize
            resized_img = self.downsample(img)
            
            # Save the resized image if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                cv2.imwrite(str(output_path), resized_img)
            
            if return_image:
                return resized_img
            return None
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_directory(self, input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.pgm'],
                         recursive: bool = True,
                         skip_existing: bool = True) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save resized images
            file_extensions: List of file extensions to process
            recursive: Whether to process subdirectories
            skip_existing: Whether to skip existing output files
            
        Returns:
            Number of processed images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        processed_images = 0
        
        # Find all image files
        all_files = []
        for ext in file_extensions:
            if recursive:
                all_files.extend(list(input_dir.glob(f"**/*{ext}")))
            else:
                all_files.extend(list(input_dir.glob(f"*{ext}")))
        
        # Process each file
        for file_path in tqdm(all_files, desc="Downsampling images", unit="file"):
            # Compute relative path to maintain directory structure
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Skip when resized image already exists
            if skip_existing and output_path.exists():
                continue
            
            # Ensure output directory exists
            os.makedirs(output_path.parent, exist_ok=True)
            
            if self.process_image(file_path, output_path) is not None:
                processed_images += 1
        
        return processed_images


def main():
    """Main function to handle command-line interface."""
    # Define interpolation method choices
    interp_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    parser = argparse.ArgumentParser(
        description="Downsample images to a target size."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input image file or directory")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output image file or directory")
    parser.add_argument("--width", type=int, default=100,
                        help="Width of output images (default: 100)")
    parser.add_argument("--height", type=int, default=100,
                        help="Height of output images (default: 100)")
    parser.add_argument("--interpolation", choices=list(interp_methods.keys()), 
                        default="area",
                        help="Interpolation method (default: area)")
    parser.add_argument("--preserve-aspect", action="store_true",
                        help="Preserve aspect ratio during resizing")
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert images to grayscale")
    parser.add_argument("--extensions", nargs="+", 
                        default=['.jpg', '.jpeg', '.png', '.pgm'],
                        help="File extensions to process (default: .jpg .jpeg .png .pgm)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't process subdirectories")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Don't skip existing output files")
    
    args = parser.parse_args()
    
    # Initialize downsampler
    downsampler = ImageDownsampler(
        target_size=(args.width, args.height),
        interpolation=interp_methods[args.interpolation],
        preserve_aspect_ratio=args.preserve_aspect,
        grayscale=args.grayscale
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input is a directory or a file
    if input_path.is_dir():
        processed = downsampler.process_directory(
            input_path,
            output_path,
            file_extensions=args.extensions,
            recursive=not args.no_recursive,
            skip_existing=not args.no_skip_existing
        )
        print(f"Processing complete! Downsampled {processed} images.")
    else:
        # Process single file
        result = downsampler.process_image(input_path, output_path, return_image=True)
        if result is not None:
            print(f"Successfully downsampled {input_path.name}. Output saved to {output_path}")
        else:
            print(f"Failed to process {input_path}")

if __name__ == "__main__":
    main()
