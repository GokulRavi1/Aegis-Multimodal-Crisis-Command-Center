import cv2
import numpy as np

# Create a short dummy video for testing purposes
video_path = "flood_footage.mp4"
fps = 30
duration_sec = 30  # 30 seconds of video = 6 frames at 5-sec intervals
width, height = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for i in range(fps * duration_sec):
    # Create a blue-ish frame simulating "flood water"
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some blue gradient
    frame[:, :, 0] = 139  # Blue channel
    frame[:, :, 1] = 69   # Green channel
    frame[:, :, 2] = 19   # Red channel (dark blue overall)
    
    # Add some text to identify the frame
    cv2.putText(frame, f"FLOOD SIMULATION - Frame {i}", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print(f"Created test video: {video_path}")
