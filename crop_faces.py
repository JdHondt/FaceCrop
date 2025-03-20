import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def crop_face(image, target_size=(192, 168)):
    """Detect and crop the face from the image and resize to target size"""
    h, w = image.shape
    
    # Load the pre-trained face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Yale dataset has challenging lighting conditions, so adjust parameters
    # Apply histogram equalization to improve face detection in extreme lighting
    equalized_img = cv2.equalizeHist(image)
    
    # Try to detect the face using cascade classifier
    # Reduce minNeighbors for better detection in difficult lighting
    faces = face_cascade.detectMultiScale(equalized_img, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    
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
        
        # Add some margin around the face (20%)
        margin_x = int(fw * 0.2)
        margin_y = int(fh * 0.2)
        
        # Ensure the face bounding box with margins stays within image bounds
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(w, x + fw + margin_x)
        y_end = min(h, y + fh + margin_y)
        
        # Crop the face with margins
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Resize the face region to the target size
        resized_face = cv2.resize(face_roi, target_size)
        return resized_face, True
    
    else:
        # Fallback method if no face is detected
        # print("  - No face detected, using fallback center crop method")
        
        # Calculate the crop dimensions to maintain aspect ratio
        aspect_ratio = target_size[0] / target_size[1]  # width/height
        
        # For the Yale dataset, we know faces are mostly centered
        # Empirically determined crop position (centered horizontally, upper-middle vertically)
        center_x = w // 2
        center_y = h // 3  # Face is usually in the upper portion of the image
        
        # Calculate crop dimensions
        if w/h > aspect_ratio:  # If image is wider than our target aspect ratio
            new_w = int(h * aspect_ratio)
            # Crop from the center
            left = max(0, center_x - new_w // 2)
            right = min(w, left + new_w)
            # Adjust if we're at the edge
            if right == w:
                left = max(0, w - new_w)
            cropped = image[:, left:right]
        else:  # If image is taller than our target aspect ratio
            new_h = int(w / aspect_ratio)
            # Crop from the upper-middle area where faces usually are
            top = max(0, center_y - new_h // 3)  # Place face at upper third
            bottom = min(h, top + new_h)
            # Adjust if we're at the edge
            if bottom == h:
                top = max(0, h - new_h)
            cropped = image[top:bottom, :]
        
        # Resize to the target size
        resized_face = cv2.resize(cropped, target_size)
        return resized_face, False

def process_dataset(input_dir, output_dir):
    """Process all PGM files in the dataset"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    total_images = 0
    face_detected = 0
    
    # Iterate through all subject directories
    for subject_dir in Path(input_dir).iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("yaleB"):
            subject_name = subject_dir.name
            print(f"Processing subject {subject_name}")
            
            # Create corresponding output directory
            subject_output_dir = os.path.join(output_dir, subject_name)
            os.makedirs(subject_output_dir, exist_ok=True)

            files = list(subject_dir.glob("*.pgm"))

            # Process all PGM files in the subject directory
            for file_path in tqdm(files, desc=f"Processing {subject_name}", unit="file"):
                output_path = os.path.join(subject_output_dir, file_path.name)

                # Skip  when cropped image already exists
                if os.path.exists(output_path):
                    continue
                
                total_images += 1
                try:
                    # Read image
                    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Error reading {file_path}")
                        continue
                    
                    # Crop and resize
                    cropped_img, detected = crop_face(img)
                    if detected:
                        face_detected += 1
                    
                    # Save the cropped image
                    cv2.imwrite(output_path, cropped_img)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Processing complete! Face detected in {face_detected}/{total_images} images ({face_detected/total_images*100:.1f}%)")

def main():
    input_dir = "original"
    output_dir = "cropped"
    
    process_dataset(input_dir, output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main()
