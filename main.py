#!/usr/bin/env python3
"""
Player Re-Identification System - Main Execution Script
"""

import os
import sys
import cv2
import time
from pathlib import Path


sys.path.append(str(Path(__file__).parent / "src"))

from player_reidentifier import PlayerReIdentifier
from config import Config

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/input",
        "data/output", 
        "results",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_setup():
    """Validate that all required files exist"""
    required_files = [
        Config.MODEL_PATH,
        Config.INPUT_VIDEO
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    """Main execution function"""
    print("🏃 Player Re-Identification System")
    print("=" * 50)
    
    
    setup_directories()
    
    
    if not validate_setup():
        print("\n📋 Setup Instructions:")
        print("1. Place your YOLO model in: models/yolov11_players.pt")
        print("2. Place your video in: data/input/15sec_input_720p.mp4")
        return
    
    
    print("\n🔧 Initializing Player Re-Identification System...")
    try:
        reidentifier = PlayerReIdentifier(
            model_path=Config.MODEL_PATH,
            similarity_threshold=Config.SIMILARITY_THRESHOLD,
            max_frames_missing=Config.MAX_FRAMES_MISSING
        )
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    
    print(f"\n🎬 Processing video: {Config.INPUT_VIDEO}")
    start_time = time.time()
    
    try:
        results = reidentifier.process_video(
            Config.INPUT_VIDEO,
            Config.OUTPUT_VIDEO
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing complete!")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Total frames: {len(results)}")
        print(f"👥 Unique players detected: {reidentifier.next_player_id - 1}")
        
        
        results_file = "results/tracking_results.pkl"
        reidentifier.save_results(results, results_file)
        print(f"💾 Results saved to: {results_file}")
        
        
        generate_summary_report(results, reidentifier)
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return

def generate_summary_report(results, reidentifier):
    """Generate a summary report of the tracking results"""
    report_file = "results/tracking_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("Player Re-Identification Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Frames Processed: {len(results)}\n")
        f.write(f"Total Players Detected: {reidentifier.next_player_id - 1}\n\n")
        
        
        f.write("Player Statistics:\n")
        f.write("-" * 20 + "\n")
        
        for player_id in range(1, reidentifier.next_player_id):
            frames_appeared = sum(1 for frame_results in results 
                                if any(pid == player_id for pid, _, _ in frame_results))
            f.write(f"Player {player_id}: Appeared in {frames_appeared} frames\n")
        
        f.write(f"\nDetailed frame-by-frame results:\n")
        f.write("-" * 30 + "\n")
        
        for frame_idx, frame_results in enumerate(results):
            if frame_results:
                player_ids = [pid for pid, _, _ in frame_results]
                f.write(f"Frame {frame_idx:3d}: Players {player_ids}\n")
    
    print(f"📄 Summary report saved to: {report_file}")

if __name__ == "__main__":
    main()