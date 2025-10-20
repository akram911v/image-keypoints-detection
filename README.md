# Image Keypoints Detection - Practical Work #2

## Detectors and Descriptors for Computer Vision

This project implements various keypoint detectors and descriptors for computer vision tasks.

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
# Single detector
python keypoint_detector.py --image path/to/image.jpg --detector SIFT

# Compare all detectors  
python keypoint_detector.py --image path/to/image.jpg --detector COMPARE
```

### Supported Detectors
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF) 
- BRISK (Binary Robust Invariant Scalable Keypoints)

### Project Files
- `keypoint_detector.py` - Main code
- `requirements.txt` - Dependencies  
- `report.md` - Technical report
