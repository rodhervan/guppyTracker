import cv2
import os

# Parameters
# video_path = 'Videos/WIN_20240306_15_11_13_Pro.mp4'
# output_folder = 'frame_set1'


video_path = 'Videos/WIN_20240306_15_21_35_Pro.mp4'
output_folder = 'frame_set2'
start_time = 30  # Start time in seconds

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to seek to a specific time in the video
def seek_to_time(cap, time_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

# Open video
cap = cv2.VideoCapture(video_path)

# Seek to start time
seek_to_time(cap, start_time)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# Parameters for evenly spaced frames
num_frames = 30
interval_seconds = duration / num_frames

# Variables for frame extraction
current_time = start_time
frame_idx = 0

# Loop to extract frames
while True:
    # Seek to current time
    seek_to_time(cap, current_time)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame
    frame_path = os.path.join(output_folder, f"frame_{frame_idx}.png")
    cv2.imwrite(frame_path, frame)
    print(f"Saved frame {frame_idx} at {current_time:.2f} seconds")
    
    # Move to next time point
    current_time += interval_seconds
    frame_idx += 1
    
    # Break if all frames extracted or end of video reached
    if frame_idx >= num_frames or current_time >= duration:
        break

# Release video capture
cap.release()
