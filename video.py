import cv2
import numpy as np
import os

def create_video_from_three_frames(frame1_path, interpolated_path, frame3_path, output_path, fps=10, duration=2):

    frame1 = cv2.imread(frame1_path)
    interpolated = cv2.imread(interpolated_path)
    frame3 = cv2.imread(frame3_path)
    
    if frame1 is None or interpolated is None or frame3 is None:
        raise ValueError("Could not read one or more of the input images")
    
    height, width = frame1.shape[:2]
    interpolated = cv2.resize(interpolated, (width, height))
    frame3 = cv2.resize(frame3, (width, height))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_per_image = int(fps * duration)
    
    for _ in range(frames_per_image):
        video.write(frame1)
    
    for _ in range(frames_per_image):
        video.write(interpolated)
    
    for _ in range(frames_per_image):
        video.write(frame3)
    
    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("video_output", exist_ok=True)
    
    # CHANGE PATHS IF NOT MATCH
    frame1_path = "test_frames/frame1.png"
    interpolated_path = "results/scene1_interpolated.png"
    frame3_path = "test_frames/frame3.png"
    
    # Output video path
    output_path = "video_output/three_frame_sequence.mp4"
    
    # Create the video
    create_video_from_three_frames(
        frame1_path, 
        interpolated_path, 
        frame3_path, 
        output_path,
        fps=30, 
        duration=1 
    )