# FaceCropper

A Python library for detecting and cropping faces from images, with support for batch processing and fallback cropping mechanisms. Built to handle challenging lighting conditions and a variety of image formats.

## Features

- Face detection using OpenCV's Haar Cascade classifier
- Fallback center cropping when face detection fails
- Configurable output image size and detection parameters
- Support for multiple image formats (PGM, JPG, JPEG, PNG)
- Batch processing of directories with recursive option
- Command-line interface and Python API
- Progress tracking with tqdm

## Background

This project was originally developed to process the Extended Yale Face Database B, a dataset containing images of faces under various lighting conditions. The dataset's images are 640x480 pixels, but the actual faces occupy a smaller region. This tool crops the images to focus only on the faces (192x168 pixels by default), which is useful for:

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
python crop_faces.py -i input_path -o output_path
```

Process a directory of images:

```bash
python crop_faces.py -i original/ -o cropped/
```

Process a single image:

```bash
python crop_faces.py -i original/yaleB11/yaleB11_P00A+000E+00.pgm -o cropped/yaleB11_P00A+000E+00.pgm
```

Customize output size:

```bash
python crop_faces.py -i original/ -o cropped/ --width 150 --height 150
```

Customize face detection parameters:

```bash
python crop_faces.py -i original/ -o cropped/ --scale-factor 1.2 --min-neighbors 5 --margin 0.3
```

### Python API

```python
from crop_faces import FaceCropper

# Initialize with default parameters
cropper = FaceCropper()

# Process a single image
cropper.process_image("path/to/image.jpg", "path/to/output.jpg")

# Process a directory of images
cropper.process_directory(
    "input_directory", 
    "output_directory",
    file_extensions=['.jpg', '.png'],
    recursive=True,
    skip_existing=True
)

# Custom parameters
custom_cropper = FaceCropper(
    target_size=(150, 150),
    min_face_size=(70, 70),
    scale_factor=1.2,
    min_neighbors=5,
    margin_percent=0.3
)
```

## Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--width` | Width of output images | 192 |
| `--height` | Height of output images | 168 |
| `--min-face-width` | Minimum face width to detect | 60 |
| `--min-face-height` | Minimum face height to detect | 60 |
| `--scale-factor` | Scale factor for face detection | 1.1 |
| `--min-neighbors` | Min neighbors parameter for face detection | 3 |
| `--margin` | Margin around detected faces (percentage) | 0.2 |
| `--extensions` | File extensions to process | .pgm .jpg .jpeg .png |
| `--no-recursive` | Don't process subdirectories | False |
| `--no-skip-existing` | Don't skip existing output files | False |

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in images.
2. **Fallback Mechanism**: When detection fails (common in extreme lighting), uses a center-based cropping strategy.
3. **Preprocessing**: Applies histogram equalization to improve detection in challenging lighting conditions.
4. **Margin Addition**: Adds configurable margins around detected faces to include more context.
5. **Resizing**: Crops and resizes images to the target dimensions.

## Original Yale Face Database B

The Extended Yale Face Database B contains 16,128 images of 28 human subjects under 9 poses and 64 illumination conditions. The dataset is widely used for facial recognition research, particularly for algorithms that need to be robust against varying lighting conditions.

## Examples

```python
# Simple example to crop a directory of images
from face_cropper import FaceCropper

cropper = FaceCropper(target_size=(192, 168))
cropper.process_directory("original", "cropped")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
