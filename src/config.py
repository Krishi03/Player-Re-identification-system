import os

class Config:
    # Model settings
    MODEL_PATH = "models/best (1).pt"
    
    # Video settings
    INPUT_VIDEO = "data/input/15sec_input_720p.mp4"
    OUTPUT_VIDEO = "data/output/tracked_output.mp4"
    
    # Re-identification parameters
    SIMILARITY_THRESHOLD = 0.7      # Adjust based on your needs
    MAX_FRAMES_MISSING = 30         # ~1 second at 30fps
    MAX_FEATURE_HISTORY = 10        # Features to keep per player
    
    # Feature extraction settings
    PLAYER_RESIZE = (64, 128)       # Standard person aspect ratio
    COLOR_HIST_BINS = 32            # Color histogram bins
    HOG_FEATURES_LIMIT = 100        # Limit HOG features for performance
    
    # Detection filtering
    CONFIDENCE_THRESHOLD = 0.5      # Minimum detection confidence
    MIN_BBOX_AREA = 500             # Minimum bounding box area
    
    # Visualization settings
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (255, 192, 203), (0, 128, 128), (128, 128, 0), (128, 0, 0)
    ]