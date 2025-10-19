import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class KeypointDetector:
    def __init__(self):
        self.detectors = {
            'SIFT': cv2.SIFT_create(),
            'ORB': cv2.ORB_create(),
            'BRISK': cv2.BRISK_create()
        }
    
    def load_image(self, image_path):
        """Load and convert image to grayscale"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray
    
    def detect_keypoints(self, image, detector_name='SIFT'):
        """Detect keypoints using specified detector"""
        if detector_name not in self.detectors:
            raise ValueError(f"Detector {detector_name} not supported. Choose from: {list(self.detectors.keys())}")
        
        detector = self.detectors[detector_name]
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors

def main():
    print("Keypoint Detector - Practical Work #2")
    print("=====================================")
    
    # Initialize detector
    detector = KeypointDetector()
    
    # Example usage (we'll add more functionality later)
    print("Available detectors:", list(detector.detectors.keys()))
    print("Basic structure ready - we'll add more features in next steps")

if __name__ == "__main__":
    main()
