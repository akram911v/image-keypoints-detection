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
    
    def visualize_keypoints(self, image, keypoints, detector_name):
        """Visualize keypoints on the image"""
        # Draw keypoints on the image
        keypoint_image = cv2.drawKeypoints(
            image, 
            keypoints, 
            None, 
            color=(0, 255, 0),  # Green color
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Convert BGR to RGB for matplotlib
        keypoint_image_rgb = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(keypoint_image_rgb)
        plt.title(f'{detector_name} Detector - {len(keypoints)} Keypoints')
        plt.axis('off')
        
        # Save the result
        output_filename = f'keypoints_{detector_name}.jpg'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Keypoints visualization saved as: {output_filename}")
        return output_filename

def main():
    print("Keypoint Detector - Practical Work #2")
    print("=====================================")
    
    # Initialize detector
    detector = KeypointDetector()
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Detect keypoints in an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--detector', type=str, choices=['SIFT', 'ORB', 'BRISK'], 
                       default='SIFT', help='Type of detector to use (default: SIFT)')
    
    args = parser.parse_args()
    
    try:
        # Load image
        image, gray = detector.load_image(args.image)
        print(f"Loaded image: {args.image}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect keypoints
        print(f"\nUsing {args.detector} detector...")
        keypoints, descriptors = detector.detect_keypoints(gray, args.detector)
        
        print(f"Number of keypoints detected: {len(keypoints)}")
        if descriptors is not None:
            print(f"Descriptor shape: {descriptors.shape}")
        
        # Visualize keypoints
        detector.visualize_keypoints(image, keypoints, args.detector)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the image path is correct and the image format is supported.")

if __name__ == "__main__":
    main()
