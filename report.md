# Practical Work #2 Report: Detectors and Descriptors

## Objective
Familiarization with keypoint detectors and descriptors, developing skills in finding key points in images using Python.

## Implemented Features

### 1. Keypoint Detectors
The project implements three popular keypoint detectors:
- **SIFT** (Scale-Invariant Feature Transform)
- **ORB** (Oriented FAST and Rotated BRIEF)
- **BRISK** (Binary Robust Invariant Scalable Keypoints)

### 2. Functionality
- **Image Loading and Preprocessing**: Automatic grayscale conversion
- **Keypoint Detection**: Multiple detector support with parameter customization
- **Visualization**: Clear visualization of detected keypoints
- **Performance Analysis**: Comparison of detector efficiency and keypoint count
- **Command-line Interface**: User-friendly parameter configuration

### 3. Technical Implementation

#### Class Structure
```python
class KeypointDetector:
    - __init__(): Initializes SIFT, ORB, and BRISK detectors
    - load_image(): Handles image loading and preprocessing
    - detect_keypoints(): Performs keypoint detection
    - visualize_keypoints(): Creates visual representations
    - compare_detectors(): Compares performance across detectors
    - plot_comparison(): Generates comparative analysis graphs
