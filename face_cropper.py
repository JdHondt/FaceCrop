#!/usr/bin/env python3
"""
BodyPartDetector: A library for detecting and cropping body parts from images

This module provides tools for detecting and cropping faces, eyes, noses, mouths, etc. from images,
with options for handling different image formats and configuring the output.
"""

import os
import cv2
import argparse
import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Union, Optional


class BodyPartDetector:
    """A class for detecting and cropping body parts from images."""
    
    def __init__(self, body_part: str = "face",
                 target_size: Tuple[int, int] = (192, 168), 
                 min_size: Tuple[int, int] = (60, 60),
                 scale_factor: float = 1.1,
                 min_neighbors: int = 3,
                 margin_percent: float = 0.2,
                 cascades_json: str = None):
        """
        Initialize the BodyPartDetector with configurable parameters.
        
        Args:
            body_part: Body part to detect ('face', 'left_eye', 'right_eye', 'nose', 'mouth')
            target_size: Output image size as (width, height)
            min_size: Minimum body part size to detect as (width, height)
            scale_factor: Scale factor for detection cascade
            min_neighbors: Minimum neighbors parameter for detection
            margin_percent: Margin to add around detected parts (as percentage of part size)
            cascades_json: Path to JSON file with cascade file mappings
        """
        self.body_part = body_part
        self.target_size = target_size
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.margin_percent = margin_percent
        
        # Load cascade mappings
        if cascades_json is None:
            cascades_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "resources/cascades.json")
        
        with open(cascades_json, 'r') as f:
            cascades = json.load(f)
        
        if body_part not in cascades:
            raise ValueError(f"Body part '{body_part}' not found in cascades.json. "
                           f"Available options: {list(cascades.keys())}")
        
        # Initialize detector
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + cascades[body_part]
        )
    
    def detect_body_part(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect and crop the specified body part from the image and resize to target size.
        
        Args:
            image: Grayscale input image as numpy array
            
        Returns:
            Tuple of (cropped_image, detected_flag)
        """
        h, w = image.shape
        
        # Apply histogram equalization to improve detection in extreme lighting
        equalized_img = cv2.equalizeHist(image)
        
        # Try to detect the body part using cascade classifier
        detections = self.cascade.detectMultiScale(
            equalized_img, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors, 
            minSize=self.min_size
        )
        
        if len(detections) > 0:
            # Get the largest detection if multiple are detected
            if len(detections) > 1:
                largest_area = 0
                largest_detection = None
                for (x, y, fw, fh) in detections:
                    if fw*fh > largest_area:
                        largest_area = fw*fh
                        largest_detection = (x, y, fw, fh)
                (x, y, fw, fh) = largest_detection
            else:
                (x, y, fw, fh) = detections[0]
            
            # Add margin around the detection
            margin_x = int(fw * self.margin_percent)
            margin_y = int(fh * self.margin_percent)
            
            # Ensure the bounding box with margins stays within image bounds
            x_start = max(0, x - margin_x)
            y_start = max(0, y - margin_y)
            x_end = min(w, x + fw + margin_x)
            y_end = min(h, y + fh + margin_y)
            
            # Crop the region with margins
            roi = image[y_start:y_end, x_start:x_end]
            
            # Resize the region to the target size
            resized = cv2.resize(roi, self.target_size)
            return resized, True
        
        else:
            # Fallback method if nothing is detected
            # Calculate the crop dimensions to maintain aspect ratio
            aspect_ratio = self.target_size[0] / self.target_size[1]  # width/height
            
            # Assume parts are mostly centered
            center_x = w // 2
            center_y = h // 3  # Usually in the upper portion of the image for face parts
            
            # Calculate crop dimensions
            if w/h > aspect_ratio:  # If image is wider than our target aspect ratio
                new_w = int(h * aspect_ratio)
                left = max(0, center_x - new_w // 2)
                right = min(w, left + new_w)
                # Adjust if we're at the edge
                if right == w:
                    left = max(0, w - new_w)
                cropped = image[:, left:right]
            else:  # If image is taller than our target aspect ratio
                new_h = int(w / aspect_ratio)
                top = max(0, center_y - new_h // 3)  # Place face/part at upper third
                bottom = min(h, top + new_h)
                # Adjust if we're at the edge
                if bottom == h:
                    top = max(0, h - new_h)
                cropped = image[top:bottom, :]
            
            # Resize to the target size
            resized = cv2.resize(cropped, self.target_size)
            return resized, False
    
    def process_image(self, image_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None,
                      return_image: bool = False) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the cropped image (if None, image is not saved)
            return_image: Whether to return the processed image
            
        Returns:
            If return_image is True, returns tuple of (cropped_image, detection_flag)
            Otherwise returns None
        """
        try:
            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Error reading {image_path}")
            
            # Detect and crop
            cropped_img, detected = self.detect_body_part(img)
            
            # Save the cropped image if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                cv2.imwrite(str(output_path), cropped_img)
            
            if return_image:
                return cropped_img, detected
            return None
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            if return_image:
                return None, False
            return None

    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         file_extensions: List[str] = ['.pgm', '.jpg', '.jpeg', '.png'],
                         recursive: bool = True,
                         skip_existing: bool = True) -> Tuple[int, int]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save cropped images
            file_extensions: List of file extensions to process
            recursive: Whether to process subdirectories
            skip_existing: Whether to skip existing output files
            
        Returns:
            Tuple of (total_processed_images, successful_detections)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        total_images = 0
        detected_count = 0
        
        # Find all image files
        if recursive:
            all_files = []
            for ext in file_extensions:
                all_files.extend(list(input_dir.glob(f"**/*{ext}")))
        else:
            all_files = []
            for ext in file_extensions:
                all_files.extend(list(input_dir.glob(f"*{ext}")))
        
        # Process each file
        for file_path in tqdm(all_files, desc=f"Processing images ({self.body_part})", unit="file"):
            # Compute relative path to maintain directory structure
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Skip when cropped image already exists
            if skip_existing and output_path.exists():
                continue
            
            # Ensure output directory exists
            os.makedirs(output_path.parent, exist_ok=True)
            
            total_images += 1
            result = self.process_image(file_path, output_path, return_image=True)
            
            if result and result[1]:  # If part was detected
                detected_count += 1
        
        return total_images, detected_count


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect and crop body parts from images."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input image file or directory")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output image file or directory")
    parser.add_argument("-p", "--part", default="face",
                        help="Body part to detect (face, left_eye, right_eye, nose, mouth)")
    parser.add_argument("-c", "--cascades", 
                        help="Path to JSON file with cascade mappings")
    parser.add_argument("--width", type=int, default=192,
                        help="Width of output images (default: 192)")
    parser.add_argument("--height", type=int, default=168,
                        help="Height of output images (default: 168)")
    parser.add_argument("--min-width", type=int, default=60,
                        help="Minimum part width to detect (default: 60)")
    parser.add_argument("--min-height", type=int, default=60,
                        help="Minimum part height to detect (default: 60)")
    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="Scale factor for detection (default: 1.1)")
    parser.add_argument("--min-neighbors", type=int, default=3,
                        help="Min neighbors parameter for detection (default: 3)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Margin around detected parts (default: 0.2)")
    parser.add_argument("--extensions", nargs="+", 
                        default=['.pgm', '.jpg', '.jpeg', '.png'],
                        help="File extensions to process (default: .pgm .jpg .jpeg .png)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't process subdirectories")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Don't skip existing output files")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BodyPartDetector(
        body_part=args.part,
        target_size=(args.width, args.height),
        min_size=(args.min_width, args.min_height),
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        margin_percent=args.margin,
        cascades_json=args.cascades
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input is a directory or a file
    if input_path.is_dir():
        total, detected = detector.process_directory(
            input_path,
            output_path,
            file_extensions=args.extensions,
            recursive=not args.no_recursive,
            skip_existing=not args.no_skip_existing
        )
        print(f"Processing complete! {args.part} detected in {detected}/{total} images "
              f"({detected/total*100 if total > 0 else 0:.1f}%)")
    else:
        # Process single file
        result = detector.process_image(input_path, output_path, return_image=True)
        if result:
            _, detected = result
            status = "detected" if detected else "not detected"
            print(f"{args.part.capitalize()} {status} in {input_path.name}. Output saved to {output_path}")
        else:
            print(f"Failed to process {input_path}")


if __name__ == "__main__":
    # Dry run mode
    if len(sys.argv) == 1:
        sys.argv = [
            "face_cropper.py",
            "-i", "face_cropped",
            "-o", "nose_cropped",
            "--width", "38",
            "--height", "38",
            "--part", "nose",
            "--min-width", "10",
            "--min-height", "10",
        ]




    main()
