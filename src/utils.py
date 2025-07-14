#!/usr/bin/env python3
"""
Utility functions for Player Re-Identification System
Contains helper functions for video processing, feature extraction, and analysis
"""

import cv2
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import time
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Utility class for video processing operations"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Get comprehensive video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'codec': ''.join([chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]),
            'bitrate': cap.get(cv2.CAP_PROP_BITRATE) if hasattr(cv2, 'CAP_PROP_BITRATE') else 'N/A'
        }
        
        cap.release()
        return info
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every nth frame
            
        Returns:
            List of saved frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {frame_count} total frames")
        return frame_paths
    
    @staticmethod
    def create_video_from_frames(frame_paths: List[str], output_path: str, fps: float = 30.0):
        """
        Create video from list of frame paths
        
        Args:
            frame_paths: List of frame image paths
            output_path: Output video path
            fps: Frames per second
        """
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        
        out.release()
        logger.info(f"Created video: {output_path}")

class FeatureUtils:
    """Utility functions for feature extraction and processing"""
    
    @staticmethod
    def compute_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Compute color histogram for BGR image
        
        Args:
            image: BGR image
            bins: Number of bins per channel
            
        Returns:
            Concatenated histogram for all channels
        """
        hist_features = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist_features.extend(hist.flatten())
        
        return np.array(hist_features)
    
    @staticmethod
    def compute_hog_features(image: np.ndarray, orientations: int = 9, 
                           pixels_per_cell: Tuple[int, int] = (8, 8),
                           cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Compute HOG features for image
        
        Args:
            image: Grayscale image
            orientations: Number of orientation bins
            pixels_per_cell: Size of each cell
            cells_per_block: Number of cells per block
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's HOG descriptor
        hog = cv2.HOGDescriptor(
            _winSize=(image.shape[1] // pixels_per_cell[1] * pixels_per_cell[1],
                     image.shape[0] // pixels_per_cell[0] * pixels_per_cell[0]),
            _blockSize=(pixels_per_cell[1] * cells_per_block[1],
                       pixels_per_cell[0] * cells_per_block[0]),
            _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
            _cellSize=(pixels_per_cell[1], pixels_per_cell[0]),
            _nbins=orientations
        )
        
        features = hog.compute(image)
        return features.flatten() if features is not None else np.array([])
    
    @staticmethod
    def compute_lbp_features(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Compute Local Binary Pattern features
        
        Args:
            image: Grayscale image
            radius: Radius of sample points
            n_points: Number of sample points
            
        Returns:
            LBP histogram
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP implementation
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                        if image[x, y] > center:
                            pattern |= (1 << k)
                lbp[i, j] = pattern
        
        # Compute histogram
        hist, _ = np.histogram(lbp, bins=2**n_points, range=(0, 2**n_points))
        return hist.astype(np.float32)
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector
        
        Args:
            features: Feature vector
            
        Returns:
            Normalized feature vector
        """
        if len(features) == 0:
            return features
        
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features

class BoundingBoxUtils:
    """Utility functions for bounding box operations"""
    
    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """
        Compute center point of bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            (center_x, center_y)
        """
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    @staticmethod
    def compute_bbox_area(bbox: List[float]) -> float:
        """
        Compute area of bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Area in pixels
        """
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    @staticmethod
    def expand_bbox(bbox: List[float], expand_ratio: float = 0.1) -> List[float]:
        """
        Expand bounding box by given ratio
        
        Args:
            bbox: [x1, y1, x2, y2]
            expand_ratio: Ratio to expand
            
        Returns:
            Expanded bounding box
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        expand_w = width * expand_ratio
        expand_h = height * expand_ratio
        
        return [
            bbox[0] - expand_w,
            bbox[1] - expand_h,
            bbox[2] + expand_w,
            bbox[3] + expand_h
        ]

class ResultsAnalyzer:
    """Utility class for analyzing tracking results"""
    
    def __init__(self, results_path: str):
        """
        Initialize analyzer with results file
        
        Args:
            results_path: Path to results pickle file
        """
        with open(results_path, 'rb') as f:
            self.data = pickle.load(f)
        self.results = self.data['results']
        self.player_features = self.data.get('player_features', {})
        self.player_positions = self.data.get('player_positions', {})
    
    def get_player_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for each player
        
        Returns:
            Dictionary with player statistics
        """
        stats = {}
        
        for frame_idx, frame_results in enumerate(self.results):
            for player_id, bbox, confidence in frame_results:
                if player_id not in stats:
                    stats[player_id] = {
                        'frames_appeared': 0,
                        'total_confidence': 0.0,
                        'first_appearance': frame_idx,
                        'last_appearance': frame_idx,
                        'bbox_areas': [],
                        'positions': []
                    }
                
                stats[player_id]['frames_appeared'] += 1
                stats[player_id]['total_confidence'] += confidence
                stats[player_id]['last_appearance'] = frame_idx
                
                area = BoundingBoxUtils.compute_bbox_area(bbox)
                stats[player_id]['bbox_areas'].append(area)
                
                center = BoundingBoxUtils.compute_bbox_center(bbox)
                stats[player_id]['positions'].append(center)
        
        # Calculate derived statistics
        for player_id in stats:
            stats[player_id]['avg_confidence'] = (
                stats[player_id]['total_confidence'] / 
                stats[player_id]['frames_appeared']
            )
            stats[player_id]['avg_bbox_area'] = np.mean(stats[player_id]['bbox_areas'])
            stats[player_id]['duration'] = (
                stats[player_id]['last_appearance'] - 
                stats[player_id]['first_appearance'] + 1
            )
        
        return stats
    
    def plot_player_trajectories(self, output_path: str = 'player_trajectories.png'):
        """
        Plot player movement trajectories
        
        Args:
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        stats = self.get_player_statistics()
        colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))
        
        for i, (player_id, player_stats) in enumerate(stats.items()):
            positions = player_stats['positions']
            if len(positions) > 1:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                plt.plot(x_coords, y_coords, 'o-', color=colors[i], 
                        label=f'Player {player_id}', markersize=3, linewidth=2)
                
                # Mark start and end
                plt.plot(x_coords[0], y_coords[0], 's', color=colors[i], 
                        markersize=8, markeredgecolor='black', markeredgewidth=1)
                plt.plot(x_coords[-1], y_coords[-1], '^', color=colors[i], 
                        markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Player Movement Trajectories')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved trajectory plot to: {output_path}")
    
    def plot_player_confidence_over_time(self, output_path: str = 'player_confidence.png'):
        """
        Plot player detection confidence over time
        
        Args:
            output_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        # Collect confidence data
        player_confidences = {}
        for frame_idx, frame_results in enumerate(self.results):
            for player_id, bbox, confidence in frame_results:
                if player_id not in player_confidences:
                    player_confidences[player_id] = {'frames': [], 'confidences': []}
                player_confidences[player_id]['frames'].append(frame_idx)
                player_confidences[player_id]['confidences'].append(confidence)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(player_confidences)))
        
        for i, (player_id, data) in enumerate(player_confidences.items()):
            plt.plot(data['frames'], data['confidences'], 'o-', 
                    color=colors[i], label=f'Player {player_id}', markersize=3)
        
        plt.xlabel('Frame Number')
        plt.ylabel('Detection Confidence')
        plt.title('Player Detection Confidence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confidence plot to: {output_path}")
    
    def generate_detailed_report(self, output_path: str = 'detailed_analysis.txt'):
        """
        Generate detailed analysis report
        
        Args:
            output_path: Path to save report
        """
        stats = self.get_player_statistics()
        
        with open(output_path, 'w') as f:
            f.write("DETAILED PLAYER RE-IDENTIFICATION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Frames Processed: {len(self.results)}\n")
            f.write(f"Total Players Detected: {len(stats)}\n\n")
            
            for player_id, player_stats in stats.items():
                f.write(f"PLAYER {player_id} ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Frames Appeared: {player_stats['frames_appeared']}\n")
                f.write(f"Duration: {player_stats['duration']} frames\n")
                f.write(f"First Appearance: Frame {player_stats['first_appearance']}\n")
                f.write(f"Last Appearance: Frame {player_stats['last_appearance']}\n")
                f.write(f"Average Confidence: {player_stats['avg_confidence']:.3f}\n")
                f.write(f"Average Bbox Area: {player_stats['avg_bbox_area']:.1f} pixels\n")
                
                # Calculate movement statistics
                if len(player_stats['positions']) > 1:
                    distances = []
                    for i in range(1, len(player_stats['positions'])):
                        dist = np.linalg.norm(
                            np.array(player_stats['positions'][i]) - 
                            np.array(player_stats['positions'][i-1])
                        )
                        distances.append(dist)
                    
                    f.write(f"Average Movement per Frame: {np.mean(distances):.1f} pixels\n")
                    f.write(f"Total Distance Traveled: {sum(distances):.1f} pixels\n")
                    f.write(f"Max Movement per Frame: {max(distances):.1f} pixels\n")
                
                f.write("\n")
        
        logger.info(f"Saved detailed report to: {output_path}")

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_dir(directory: str):
        """Ensure directory exists"""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Any, filepath: str):
        """Save data to JSON file"""
        FileUtils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_file_size(filepath: str) -> str:
        """Get human-readable file size"""
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

class PerformanceMonitor:
    """Utility class for monitoring performance"""
    
    def __init__(self):
        self.start_time = None
        self.timings = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.timings[name] = {'start': time.time(), 'end': None}
    
    def end_timer(self, name: str):
        """End timing an operation"""
        if name in self.timings:
            self.timings[name]['end'] = time.time()
    
    def get_duration(self, name: str) -> float:
        """Get duration of an operation"""
        if name in self.timings and self.timings[name]['end']:
            return self.timings[name]['end'] - self.timings[name]['start']
        return 0.0
    
    def get_report(self) -> str:
        """Get performance report"""
        report = "PERFORMANCE REPORT\n"
        report += "=" * 30 + "\n"
        
        for name, timing in self.timings.items():
            if timing['end']:
                duration = timing['end'] - timing['start']
                report += f"{name}: {duration:.3f} seconds\n"
        
        return report

# Example usage and testing functions
def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test video info
    try:
        # This would work with an actual video file
        # info = VideoProcessor.get_video_info("test_video.mp4")
        # print(f"Video info: {info}")
        pass
    except Exception as e:
        print(f"Video test skipped: {e}")
    
    # Test feature extraction
    test_image = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    
    color_hist = FeatureUtils.compute_color_histogram(test_image)
    print(f"Color histogram shape: {color_hist.shape}")
    
    hog_features = FeatureUtils.compute_hog_features(test_image)
    print(f"HOG features shape: {hog_features.shape}")
    
    # Test bounding box utils
    bbox1 = [10, 10, 50, 50]
    bbox2 = [30, 30, 70, 70]
    iou = BoundingBoxUtils.compute_iou(bbox1, bbox2)
    print(f"IoU between boxes: {iou:.3f}")
    
    print("All utility tests passed!")

if __name__ == "__main__":
    test_utils()