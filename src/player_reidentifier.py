import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from scipy.spatial.distance import cosine
import pickle
import os

class PlayerReIdentifier:
    def __init__(self, model_path, similarity_threshold=0.65, max_frames_missing=90, position_weight=0.3):
        """
        Initialize the Player Re-Identification system
        
        Args:
            model_path: Path to the YOLO model
            similarity_threshold: Threshold for feature similarity matching (lowered for better recall)
            max_frames_missing: Maximum frames a player can be missing before being considered new (increased)
            position_weight: Weight for position consistency in matching
        """
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        self.max_frames_missing = max_frames_missing
        self.position_weight = position_weight
        
        # Player tracking data
        self.player_features = {}  
        self.player_last_seen = {}  
        self.player_positions = {}  
        self.player_appearance_history = {}  
        self.player_size_history = {}  
        self.next_player_id = 1
        
       
        self.frame_count = 0
        self.results_history = []
        self.disappeared_players = {} 
        
    def extract_enhanced_features(self, frame, bbox):
       
        try:
            # Validate and process bbox
            x1, y1, x2, y2 = bbox
            if not all(np.isfinite([x1, y1, x2, y2])):
                return np.zeros(768), 0  # Return features and bbox area
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure bbox is within frame boundaries
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Calculate bbox area for size consistency
            bbox_area = (x2 - x1) * (y2 - y1)
            
            # Extract player region
            player_region = frame[y1:y2, x1:x2]
            if player_region.size == 0:
                return np.zeros(768), 0
            
            # Resize to standard size
            player_region = cv2.resize(player_region, (64, 128))
            
            # Extract multiple types of features
            features = []
            
        
            hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv_region], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
           
            torso_region = player_region[int(0.2*player_region.shape[0]):int(0.8*player_region.shape[0]), :]
            torso_hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            for i in range(3):
                hist = cv2.calcHist([torso_hsv], [i], None, [16], [0, 256])
                features.extend(hist.flatten())
            
           
            gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            
           
            lbp_features = []
            for radius in [1, 2]:
                for y in range(radius, gray.shape[0]-radius):
                    for x in range(radius, gray.shape[1]-radius):
                        center = gray[y, x]
                        binary_code = 0
                        for i, (dy, dx) in enumerate([(-radius,-radius), (-radius,0), (-radius,radius), 
                                                     (0,radius), (radius,radius), (radius,0), 
                                                     (radius,-radius), (0,-radius)]):
                            if gray[y+dy, x+dx] > center:
                                binary_code |= (1 << i)
                        lbp_features.append(binary_code)
            
            # Create histogram of LBP codes
            lbp_hist, _ = np.histogram(lbp_features, bins=32, range=(0, 256))
            features.extend(lbp_hist / max(1, len(lbp_features)))
            
            # 3. Structural features (body proportions)
            # These are more stable for re-identification
            height, width = player_region.shape[:2]
            aspect_ratio = width / height
            
            # Divide into regions and get color statistics
            regions = [
                player_region[0:height//3, :],  # Head region
                player_region[height//3:2*height//3, :],  # Torso region
                player_region[2*height//3:, :]  # Legs region
            ]
            
            for region in regions:
                if region.size > 0:
                    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    mean_color = np.mean(region_hsv, axis=(0, 1))
                    std_color = np.std(region_hsv, axis=(0, 1))
                    features.extend(mean_color)
                    features.extend(std_color)
                else:
                    features.extend(np.zeros(6))
            
            # 4. Edge features (clothing boundaries)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            features.append(aspect_ratio)
            
            # Normalize features
            features = np.array(features)
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            return features, bbox_area
            
        except Exception as e:
            print(f"Error in extract_enhanced_features: {e}")
            return np.zeros(768), 0
    
    def calculate_enhanced_similarity(self, features1, features2, size1=None, size2=None):
        """Enhanced similarity calculation with size consistency"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        try:
            # Feature similarity
            feature_sim = 1 - cosine(features1, features2)
            
            # Size consistency (if available)
            if size1 is not None and size2 is not None and size1 > 0 and size2 > 0:
                size_ratio = min(size1, size2) / max(size1, size2)
                # Penalize if size difference is too large
                if size_ratio < 0.5:  # More than 2x size difference
                    feature_sim *= 0.7
                elif size_ratio < 0.7:  # Moderate size difference
                    feature_sim *= 0.9
            
            return feature_sim
        except:
            return 0.0
    
    def find_best_match(self, current_features, current_position, current_size):
        """
        Enhanced matching with multiple criteria
        """
        best_match = None
        best_score = 0
        
        # Check active players first
        for player_id, feature_history in self.player_features.items():
            frames_missing = self.frame_count - self.player_last_seen.get(player_id, 0)
            
            # Skip if player has been missing too long
            if frames_missing > self.max_frames_missing:
                continue
            
            # Calculate feature similarity
            similarities = []
            recent_features = list(feature_history)[-10:]  # Use more history
            recent_sizes = list(self.player_size_history.get(player_id, deque()))[-10:]
            
            for i, features in enumerate(recent_features):
                size = recent_sizes[i] if i < len(recent_sizes) else None
                sim = self.calculate_enhanced_similarity(current_features, features, current_size, size)
                similarities.append(sim)
            
            if not similarities:
                continue
                
            # Weight recent observations more heavily
            weights = np.exp(np.linspace(-1, 0, len(similarities)))
            avg_similarity = np.average(similarities, weights=weights)
            
            # Position consistency bonus
            position_score = 1.0
            if player_id in self.player_positions and self.player_positions[player_id]:
                recent_positions = list(self.player_positions[player_id])[-5:]
                
                # Calculate expected position based on recent movement
                if len(recent_positions) >= 2:
                    # Predict next position based on velocity
                    last_pos = np.array(recent_positions[-1])
                    prev_pos = np.array(recent_positions[-2])
                    velocity = last_pos - prev_pos
                    predicted_pos = last_pos + velocity * frames_missing
                    
                    pos_distance = np.linalg.norm(np.array(current_position) - predicted_pos)
                else:
                    pos_distance = np.linalg.norm(np.array(current_position) - np.array(recent_positions[-1]))
                
                # Position scoring
                if pos_distance < 50:  # Very close
                    position_score = 1.2
                elif pos_distance < 100:  # Close
                    position_score = 1.0
                elif pos_distance < 200:  # Moderate distance
                    position_score = 0.9
                else:  # Far
                    position_score = 0.7
            
            # Combine similarity and position
            final_score = avg_similarity * position_score
            
            # Bonus for recently seen players
            if frames_missing < 10:
                final_score *= 1.1
            elif frames_missing < 30:
                final_score *= 1.05
            
            if final_score > best_score and avg_similarity > self.similarity_threshold:
                best_score = final_score
                best_match = player_id
        
        return best_match
    
    def update_player_data(self, player_id, features, position, size):
        """Update player's feature history and position"""
        if player_id not in self.player_features:
            self.player_features[player_id] = deque(maxlen=15)  # Keep more history
            self.player_positions[player_id] = deque(maxlen=15)
            self.player_size_history[player_id] = deque(maxlen=15)
        
        self.player_features[player_id].append(features)
        self.player_positions[player_id].append(position)
        self.player_size_history[player_id].append(size)
        self.player_last_seen[player_id] = self.frame_count
    
    def process_frame(self, frame):
        """
        Process a single frame and return player identifications
        """
        # Run YOLO detection
        results = self.model(frame)
        
        current_detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls) == 0:  # Person class
                        # Get bbox coordinates safely
                        bbox_tensor = box.xyxy[0].cpu().numpy()
                        bbox = bbox_tensor.astype(float)
                        
                        # Get confidence safely
                        confidence_tensor = box.conf[0].cpu().numpy()
                        confidence = float(confidence_tensor)
                        
                        # Skip low confidence detections
                        if confidence < 0.5:
                            continue
                        
                        # Validate bbox
                        if not all(np.isfinite(bbox)):
                            continue
                        
                        # Extract enhanced features
                        features, bbox_area = self.extract_enhanced_features(frame, bbox)
                        
                        # Get center position
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        position = [center_x, center_y]
                        
                        current_detections.append((bbox, confidence, features, position, bbox_area))
        
        # Sort detections by confidence (process high-confidence detections first)
        current_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Match detections to existing players
        frame_results = []
        used_player_ids = set()
        
        for bbox, confidence, features, position, bbox_area in current_detections:
            # Try to match with existing player
            matched_player_id = self.find_best_match(features, position, bbox_area)
            
            # Ensure we don't assign the same player ID to multiple detections in one frame
            if matched_player_id is not None and matched_player_id not in used_player_ids:
                # Update existing player
                self.update_player_data(matched_player_id, features, position, bbox_area)
                player_id = matched_player_id
                used_player_ids.add(player_id)
            else:
                # Create new player
                player_id = self.next_player_id
                self.next_player_id += 1
                self.update_player_data(player_id, features, position, bbox_area)
                used_player_ids.add(player_id)
            
            frame_results.append((player_id, bbox, confidence))
        
        self.frame_count += 1
        self.results_history.append(frame_results)
        
        return frame_results
    
    def process_video(self, video_path, output_path=None):
        """
        Process entire video and return all results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_results = self.process_frame(frame)
            all_results.append(frame_results)
            
            # Draw results on frame
            if output_path:
                annotated_frame = self.draw_results(frame, frame_results)
                out.write(annotated_frame)
            
            # Print progress
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processed {self.frame_count} frames ({progress:.1f}%)")
        
        cap.release()
        if output_path:
            out.release()
        
        return all_results
    
    def draw_results(self, frame, results):
        """Draw bounding boxes and player IDs on frame with enhanced visualization"""
        annotated_frame = frame.copy()
        
        # Define more distinct colors for different players
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 192, 203), (0, 128, 128),
            (255, 20, 147), (0, 191, 255), (255, 69, 0), (50, 205, 50), (138, 43, 226)
        ]
        
        for player_id, bbox, confidence in results:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get color for this player
                color = colors[player_id % len(colors)]
                
                # Draw bounding box with thickness based on confidence
                thickness = max(2, int(confidence * 4))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw player ID and confidence
                label = f"ID:{player_id} ({confidence:.2f})"
                font_scale = 0.7
                font_thickness = 2
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Draw label background
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), 
                             color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
                
            except Exception as e:
                print(f"Error drawing bbox: {e}")
                continue
        
        return annotated_frame
    
    def save_results(self, results, output_file):
        """Save results to file"""
        with open(output_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'player_features': dict(self.player_features),
                'player_positions': dict(self.player_positions),
                'player_size_history': dict(self.player_size_history)
            }, f)
    
    def load_results(self, input_file):
        """Load results from file"""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
            return data['results']
    
    def get_tracking_statistics(self):
        """Get detailed tracking statistics"""
        stats = {
            'total_players': self.next_player_id - 1,
            'frames_processed': self.frame_count,
            'player_frame_counts': {}
        }
        
        # Count frames each player appeared in
        for player_id in range(1, self.next_player_id):
            frame_count = 0
            for frame_results in self.results_history:
                if any(pid == player_id for pid, _, _ in frame_results):
                    frame_count += 1
            stats['player_frame_counts'][player_id] = frame_count
        
        return stats

# Usage example
if __name__ == "__main__":
    # Initialize the re-identifier with optimized parameters
    model_path = "models/yolov11_players.pt"
    video_path = "data/input/15sec_input_720p.mp4"
    
    reidentifier = PlayerReIdentifier(
        model_path,
        similarity_threshold=0.65,  # Lowered for better recall
        max_frames_missing=90,     # Increased for longer re-identification
        position_weight=0.3
    )
    
    # Process the video
    print("Processing video...")
    results = reidentifier.process_video(video_path, "data/output/output_with_tracking.mp4")
    
    # Save results
    reidentifier.save_results(results, "results/player_tracking_results.pkl")
    
    # Get statistics
    stats = reidentifier.get_tracking_statistics()
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {stats['frames_processed']}")
    print(f"Total unique players detected: {stats['total_players']}")
    
    # Print per-player statistics
    print("\nPlayer appearance statistics:")
    for player_id, frame_count in stats['player_frame_counts'].items():
        percentage = (frame_count / stats['frames_processed']) * 100
        print(f"Player {player_id}: {frame_count} frames ({percentage:.1f}%)")