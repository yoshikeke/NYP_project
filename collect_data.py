import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
MODEL_PATH = "C:/dev/litter_segmentation/runs/segment/train3/weights/best.pt"
VIDEO_PATH = r"C:\dev\litter_segmentation\data\throwing\HRL -80m_Camera 01_20240325162243_20240325162801.mp4"
CONFIDENCE_THRESHOLD = 0.5

# --- Main Logic ---

# 1. Load the YOLOv8 Model
# This single line handles loading the model and setting it to the correct device (GPU or CPU)
print("Loading YOLOv8 model...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Load the Video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    exit()

# 3. Process Video Frame by Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # A. Run Inference
    # The model handles all pre-processing automatically!
    results = model(frame, verbose=False) # verbose=False suppresses console output

    # B. Post-process and Visualize
    # The 'results' object contains all the detections, masks, etc.
    overlay = frame.copy()

    # Check if there are any masks in the results
    if results[0].masks is not None:
        # Iterate through each detected object's mask
        for mask in results[0].masks:
            # The mask.xy contains the contour points of the segmented object
            contour = mask.xy[0]
            contour = np.array(contour, dtype=np.int32) # Convert to integer numpy array

            # Draw the filled contour on the overlay
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)

    # Blend the overlay with the original frame
    alpha = 0.4 # Transparency factor
    result_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # C. Display the result
    cv2.imshow('YOLOv8 Segmentation', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()