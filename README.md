# Player Re-Identification System

## Overview
This project is a Player Re-Identification System designed to track and re-identify players in sports videos. It uses YOLO for player detection and advanced appearance-based feature extraction to assign consistent IDs to players, even if they leave and re-enter the frame. The system is robust to occlusions and changes in player appearance, making it suitable for sports analytics, broadcast enhancement, and player tracking research.

## Unique Features
- **Re-Identification Across Frames:** Assigns the same ID to a player even if they leave and re-enter the frame, using appearance features and position consistency.
- **Enhanced Feature Extraction:** Combines color histograms, texture, and structural features for robust player matching.
- **Flexible Parameters:** Easily tune similarity threshold, missing frame tolerance, and feature extraction settings.
- **Automatic Directory Setup:** Creates all necessary folders for input, output, models, and results.
- **Summary Reporting:** Generates detailed statistics and summary reports after processing.

## Requirements
- Python 3.7+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (for detection)
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- torch (PyTorch)
- matplotlib (for analysis utilities)



## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare directories**
   The script will auto-create these, but you can check:
   - `data/input/` — Place your input videos here
   - `data/output/` — Output videos will be saved here
   - `models/` — Place your YOLO model here (e.g., `yolov11_players.pt` or `best.pt`)
   - `results/` — Tracking results and reports

4. **Add your model and video**
   - Place your YOLO model file in `models/` 
   - Place your input video in `data/input/` 

5. **Configure settings**
   - Edit `src/config.py` to adjust model path, thresholds, and other parameters.

## Usage
Run the main script:
```bash
python main.py
```

- The script will:
  - Check for required files and directories
  - Initialize the system
  - Process the video, saving an annotated output video and results
  - Print and save summary statistics

### Output
- Annotated video: `data/output/tracked_output.mp4` (or as configured)
- Results: `results/tracking_results.pkl` and `results/tracking_summary.txt`
