# BodyPartDetector

A Python library for detecting and cropping body parts (faces, eyes, nose, mouth) from images, with support for batch processing and fallback cropping mechanisms. Built to handle challenging lighting conditions and a variety of image formats.

## Features

- Detection of faces and facial features using OpenCV's Haar Cascade classifiers
- Support for multiple body parts (face, left_eye, right_eye, nose, mouth)
- Fallback center cropping when detection fails
- Configurable output image size and detection parameters
- Support for multiple image formats (PGM, JPG, JPEG, PNG)
- Batch processing of directories with recursive option
- Command-line interface and Python API
- Progress tracking with tqdm

## Background

This project was originally developed to process the Extended Yale Face Database B, a dataset containing images of faces under various lighting conditions. The dataset's images are 640x480 pixels, but the actual faces occupy a smaller region. This tool crops the images to focus only on specific body parts (192x168 pixels by default for faces), which is useful for:

- Reducing storage requirements
- Focusing machine learning models on relevant features
- Standardizing input size for computer vision tasks
- Improving processing efficiency

## Installation

### Requirements

- Python 3.6+
- OpenCV
- NumPy
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:

```bash
python face_cropper.py -i input_path -o output_path
```

Process a directory of images for face detection (default):

```bash
python face_cropper.py -i original/ -o cropped/
```

Process a directory for a specific body part:

```bash
python face_cropper.py -i original/ -o nose_cropped/ -p nose
python face_cropper.py -i original/ -o eyes_cropped/ -p left_eye
```

Process a single image:

```bash
python face_cropper.py -i original/yaleB11/yaleB11_P00A+000E+00.pgm -o cropped/yaleB11_P00A+000E+00.pgm
```

Customize output size:

```bash
python face_cropper.py -i original/ -o cropped/ --width 150 --height 150
```

Customize detection parameters:

```bash
python face_cropper.py -i original/ -o cropped/ --scale-factor 1.2 --min-neighbors 5 --margin 0.3
```

### Python API

```python
from face_cropper import BodyPartDetector

# Initialize with default parameters for face detection
detector = BodyPartDetector()

# Initialize for nose detection
nose_detector = BodyPartDetector(body_part="nose", target_size=(38, 38))

# Process a single image
detector.process_image("path/to/image.jpg", "path/to/output.jpg")

# Process a directory of images
detector.process_directory(
    "input_directory", 
    "output_directory",
    file_extensions=['.jpg', '.png'],
    recursive=True,
    skip_existing=True
)

# Custom parameters
custom_detector = BodyPartDetector(
    body_part="face",
    target_size=(150, 150),
    min_size=(70, 70),
    scale_factor=1.2,
    min_neighbors=5,
    margin_percent=0.3
)
```

## Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-p, --part` | Body part to detect (face, left_eye, right_eye, nose, mouth) | face |
| `-c, --cascades` | Path to JSON file with cascade mappings | resources/cascades.json |
| `--width` | Width of output images | 192 |
| `--height` | Height of output images | 168 |
| `--min-width` | Minimum part width to detect | 60 |
| `--min-height` | Minimum part height to detect | 60 |
| `--scale-factor` | Scale factor for detection | 1.1 |
| `--min-neighbors` | Min neighbors parameter for detection | 3 |
| `--margin` | Margin around detected parts (percentage) | 0.2 |
| `--extensions` | File extensions to process | .pgm .jpg .jpeg .png |
| `--no-recursive` | Don't process subdirectories | False |
| `--no-skip-existing` | Don't skip existing output files | False |

## How It Works

1. **Body Part Detection**: Uses OpenCV's Haar Cascade classifiers to detect faces or facial features in images.
2. **Fallback Mechanism**: When detection fails (common in extreme lighting), uses a center-based cropping strategy.
3. **Preprocessing**: Applies histogram equalization to improve detection in challenging lighting conditions.
4. **Margin Addition**: Adds configurable margins around detected regions to include more context.
5. **Resizing**: Crops and resizes images to the target dimensions.

## Available Body Parts

The detector supports the following body parts through cascades.json configuration:

- face: Whole face detection
- left_eye: Left eye detection
- right_eye: Right eye detection
- nose: Nose detection
- mouth: Mouth detection

## Original Yale Face Database B

The Extended Yale Face Database B contains 16,128 images of 28 human subjects under 9 poses and 64 illumination conditions. The dataset is widely used for facial recognition research, particularly for algorithms that need to be robust against varying lighting conditions.

## Examples

```python
# Simple example to crop faces from a directory of images
from face_cropper import BodyPartDetector

detector = BodyPartDetector(body_part="face", target_size=(192, 168))
detector.process_directory("original", "cropped_faces")

# Crop noses from the same directory with a smaller output size
nose_detector = BodyPartDetector(body_part="nose", target_size=(38, 38))
nose_detector.process_directory("original", "cropped_noses")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
