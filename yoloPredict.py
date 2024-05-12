from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from datetime import datetime
import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import time


model_path = 'Train_data/segment/train4/weights/best.pt'
video_path = 'Videos/Pez3_con_campo.mp4'

# Initialize YOLO model
model = YOLO(model_path)

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Set the start time to 90 seconds (90000 milliseconds)
start_time_ms = 240000
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate time between frames
time_difference = 1 / fps  # At constant frame rate

# Initialize a list to store dictionaries for each frame
frames_data = []

# Initialize variables for previous centroid and time
prev_centroid = None
prev_time = None

# Process each frame in the video
for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    if not ret:
        break
    
    timer = cap.get(cv2.CAP_PROP_POS_MSEC)
    seconds = timer / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int((seconds - int(seconds)) * 1000):03d}"
    # Convert frame to PIL image for YOLO processing
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Get results from YOLO model
    results = model(img_pil)
    result = results[0]
    masks = result.masks

    # Initialize dictionary for current frame
    frame_data = {
        'frame_number': frame_num,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'),
        'video_timestamp': formatted_time,
        'detected': masks is not None,
        'centroids': [],
        'velocity': [],
        'distance': 0  
    }

    # Check there are masks in the current frame to avoid errors
    if masks is not None:
        overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for mask in masks:
            polygon = mask.xy[0]
            if len(polygon) >= 3:
                overlay_draw.polygon(polygon, outline=(0, 255, 0), fill=(0, 255, 0, 127))

                polygon_shapely = Polygon(polygon)
                centroid = polygon_shapely.centroid
                frame_data['centroids'].append((centroid.x, centroid.y))  # Append centroid coordinates to the list
                circle_radius = 5
                left_up_point = (centroid.x - circle_radius, centroid.y - circle_radius)
                right_down_point = (centroid.x + circle_radius, centroid.y + circle_radius)
                overlay_draw.ellipse([left_up_point, right_down_point], fill=(0, 0, 255))

                # Calculate velocity using time difference
                if prev_centroid is not None:
                    velocity_x = (centroid.x - prev_centroid[0]) / time_difference
                    velocity_y = (centroid.y - prev_centroid[1]) / time_difference
                    frame_data['velocity'] = [velocity_x, velocity_y]
                # Calculate distance between centroids in consecutive frames
                    distance = np.sqrt((centroid.x - prev_centroid[0])**2 + (centroid.y - prev_centroid[1])**2)
                    frame_data['distance'] = distance
            else:
                frame_data['detected'] = False

        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay)
        
        # Update previous centroid and time
        prev_centroid = frame_data['centroids'][-1] if frame_data['centroids'] else None
        prev_time = frame_num / fps

    # Add text overlay for frame number or timer
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 24)
    # timer = cap.get(cv2.CAP_PROP_POS_MSEC)
    # min = time.strftime('%H:%M:%S', time.gmtime(timer/1000))
    # text = f"Frame: {frame_num}"
    draw.text((10, 10), str(formatted_time), (255, 255, 255), font=font)

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    # Append frame data to frames_data list
    frames_data.append(frame_data)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# # Print frames_data
# for frame_data in frames_data:
#     print(frame_data)

os.makedirs('Generated_data', exist_ok=True)
with open('Generated_data/Fish_data.json', 'w', encoding='utf-8') as f:
    json.dump(frames_data, f, ensure_ascii=False, indent=4)
