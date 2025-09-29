import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file at '{video_path}' was not found.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open the video file at '{video_path}'.")
        return

    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frame rate: {frame_rate} FPS")
    print(f"Total frames: {frame_count}")

    # Process each frame and save a frame every 'frame_interval' seconds
    count = 0
    frame_number = 0
    saved_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the current time in the video
        current_time = count / frame_rate

        # Check if it's time to save a frame based on the interval
        if current_time >= frame_number * frame_interval:
            frame_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
            print(f"Saved {frame_filename}")
            frame_number += 1
        
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nExtraction complete. {saved_frames} frames were saved to '{output_folder}'.")

# --- Configuration ---
# Set the path to your video file
video_file = r"C:\dev\litter_segmentation\data\throwing\HRL -80m_Camera 01_20240325162243_20240325162801.mp4" 

# Set the desired output directory for the images
output_directory = r"C:\dev\litter_segmentation\images\plastic refuse"

# Set the interval (in seconds) to capture a frame
# For example, 1.0 means one frame every second.
interval = 1/30 

# Run the function
extract_frames(video_file, output_directory, interval)