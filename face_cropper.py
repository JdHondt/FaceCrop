#!/usr/bin/env python3
"""
FaceCropper: A library for detecting and cropping faces from images

This module provides tools for detecting and cropping faces from images,
with options for handling different image formats and configuring the output.
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Union, Optional


class FaceCropper:
    """A class for detecting and cropping faces from images."""
    
    def __init__(self, target_size: Tuple[int, int] = (192, 168), 
                 min_face_size: Tuple[int, int] = (60, 60),
                 scale_factor: float = 1.1,
                 min_neighbors: int = 3,
                 margin_percent: float = 0.2):
        """
        Initialize the FaceCropper with configurable parameters.
        
        Args:
            target_size: Output image size as (width, height)
            min_face_size: Minimum face size to detect as (width, height)
            scale_factor: Scale factor for face detection cascade
            min_neighbors: Minimum neighbors parameter for face detection
            margin_percent: Margin to add around detected faces (as percentage of face size)
        """
        self.target_size = target_size
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.margin_percent = margin_percent
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def crop_face(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect and crop the face from the image and resize to target size.
        
        Args:
            image: Grayscale input image as numpy array
            
        Returns:
            Tuple of (cropped_image, face_detected_flag)
        """
        h, w = image.shape
        
        # Apply histogram equalization to improve face detection in extreme lighting
        equalized_img = cv2.equalizeHist(image)
        
        # Try to detect the face using cascade classifier
        faces = self.face_cascade.detectMultiScale(
            equalized_img, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors, 
            minSize=self.min_face_size
        )
        
        if len(faces) > 0:
            # Get the largest face if multiple are detected
            if len(faces) > 1:
                largest_area = 0
                largest_face = None
                for (x, y, fw, fh) in faces:
                    if fw*fh > largest_area:
                        largest_area = fw*fh
                        largest_face = (x, y, fw, fh)
                (x, y, fw, fh) = largest_face
            else:
                (x, y, fw, fh) = faces[0]
            
            # Add margin around the face
            margin_x = int(fw * self.margin_percent)
            margin_y = int(fh * self.margin_percent)
            
            # Ensure the face bounding box with margins stays within image bounds
            x_start = max(0, x - margin_x)
            y_start = max(0, y - margin_y)
            x_end = min(w, x + fw + margin_x)
            y_end = min(h, y + fh + margin_y)
            
            # Crop the face with margins
            face_roi = image[y_start:y_end, x_start:x_end]
            
            # Resize the face region to the target size
            resized_face = cv2.resize(face_roi, self.target_size)
            return resized_face, True
        
        else:
            # Fallback method if no face is detected
            # Calculate the crop dimensions to maintain aspect ratio
            aspect_ratio = self.target_size[0] / self.target_size[1]  # width/height
            
            # Assume faces are mostly centered
            center_x = w // 2
            center_y = h // 3  # Face is usually in the upper portion of the image
            
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
                top = max(0, center_y - new_h // 3)  # Place face at upper third
                bottom = min(h, top + new_h)
                # Adjust if we're at the edge
                if bottom == h:
                    top = max(0, h - new_h)
                cropped = image[top:bottom, :]
            
            # Resize to the target size
            resized_face = cv2.resize(cropped, self.target_size)
            return resized_face, False
    
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
            If return_image is True, returns tuple of (cropped_image, face_detected_flag)
            Otherwise returns None
        """
        try:
            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Error reading {image_path}")
            
            # Crop and resize
            cropped_img, detected = self.crop_face(img)
            
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
            Tuple of (total_processed_images, successful_face_detections)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        total_images = 0
        face_detected = 0
        
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
        for file_path in tqdm(all_files, desc="Processing images", unit="file"):
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
            
            if result and result[1]:  # If face was detected
                face_detected += 1
        
        return total_images, face_detected


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect and crop faces from images."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input image file or directory")
    parser.add_argument("-o", "--output", required=True, 
                        help="Output image file or directory")
    parser.add_argument("--width", type=int, default=192,
                        help="Width of output images (default: 192)")
    parser.add_argument("--height", type=int, default=168,
                        help="Height of output images (default: 168)")
    parser.add_argument("--min-face-width", type=int, default=60,
                        help="Minimum face width to detect (default: 60)")
    parser.add_argument("--min-face-height", type=int, default=60,
                        help="Minimum face height to detect (default: 60)")
    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="Scale factor for face detection (default: 1.1)")
    parser.add_argument("--min-neighbors", type=int, default=3,
                        help="Min neighbors parameter for face detection (default: 3)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Margin around detected faces (default: 0.2)")
    parser.add_argument("--extensions", nargs="+", 
                        default=['.pgm', '.jpg', '.jpeg', '.png'],
                        help="File extensions to process (default: .pgm .jpg .jpeg .png)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't process subdirectories")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Don't skip existing output files")
    
    args = parser.parse_args()
    
    # Initialize face cropper
    cropper = FaceCropper(
        target_size=(args.width, args.height),
        min_face_size=(args.min_face_width, args.min_face_height),
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        margin_percent=args.margin
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input is a directory or a file
    if input_path.is_dir():
        total, detected = cropper.process_directory(
            input_path,
            output_path,
            file_extensions=args.extensions,
            recursive=not args.no_recursive,
            skip_existing=not args.no_skip_existing
        )
        print(f"Processing complete! Face detected in {detected}/{total} images "
              f"({detected/total*100 if total > 0 else 0:.1f}%)")
    else:
        # Process single file
        result = cropper.process_image(input_path, output_path, return_image=True)
        if result:
            _, detected = result
            status = "detected" if detected else "not detected"
            print(f"Face {status} in {input_path.name}. Output saved to {output_path}")
        else:
            print(f"Failed to process {input_path}")


if __name__ == "__main__":
    main()
