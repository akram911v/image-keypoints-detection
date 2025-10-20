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

    def compare_detectors(self, image):
        """Compare performance of all detectors on the same image"""
        print("\n" + "="*50)
        print("DETECTOR COMPARISON")
        print("="*50)
        
        results = []
        
        for detector_name in self.detectors.keys():
            print(f"\nTesting {detector_name}...")
            
            # Time the detection process
            start_time = cv2.getTickCount()
            keypoints, descriptors = self.detect_keypoints(image, detector_name)
            end_time = cv2.getTickCount()
            
            # Calculate processing time
            time_taken = (end_time - start_time) / cv2.getTickFrequency()
            
            # Store results
            result = {
                'detector': detector_name,
                'keypoints_count': len(keypoints),
                'processing_time': time_taken,
                'has_descriptors': descriptors is not None,
                'descriptor_size': descriptors.shape if descriptors is not None else None
            }
            results.append(result)
            
            print(f"  Keypoints: {result['keypoints_count']}")
            print(f"  Time: {result['processing_time']:.4f}s")
            if result['has_descriptors']:
                print(f"  Descriptor shape: {result['descriptor_size']}")
        
        return results
    
    def plot_comparison(self, results):
        """Create comparison plots"""
        # Extract data for plotting
        detectors = [r['detector'] for r in results]
        keypoints_count = [r['keypoints_count'] for r in results]
        processing_times = [r['processing_time'] for r in results]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Keypoints count
        bars1 = ax1.bar(detectors, keypoints_count, color=['blue', 'green', 'red'])
        ax1.set_title('Number of Keypoints Detected')
        ax1.set_ylabel('Keypoint Count')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 2: Processing time
        bars2 = ax2.bar(detectors, processing_times, color=['orange', 'purple', 'brown'])
        ax2.set_title('Processing Time')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('detector_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nComparison plot saved as: detector_comparison.png")

def main():
    print("Keypoint Detector - Practical Work #2")
    print("=====================================")
    
    # Initialize detector
    detector = KeypointDetector()
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Detect keypoints in an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--detector', type=str, choices=['SIFT', 'ORB', 'BRISK', 'COMPARE'], 
                       default='SIFT', help='Type of detector to use or COMPARE for all')
    parser.add_argument('--compare', action='store_true', help='Run comparison of all detectors')
    
    args = parser.parse_args()
    
    try:
        # Load image
        image, gray = detector.load_image(args.image)
        print(f"Loaded image: {args.image}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        if args.detector == 'COMPARE' or args.compare:
            # Compare all detectors
            results = detector.compare_detectors(gray)
            detector.plot_comparison(results)
        else:
            # Use single detector
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
