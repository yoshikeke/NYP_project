from ultralytics import YOLO
import cv2

# Load your trained YOLO model
MODEL_PATH = "/home/atic/repo/litter_detect/runs/detect/train38/weights/best.pt"  # Your trained weights
model = YOLO(MODEL_PATH)

# Custom classes (same order as training dataset)
CLASS_NAMES = [
    "paper tissue",
    "plastic refuse",
    "sweet_wrapper",
]

# Path to your video
VIDEO_PATH = "/home/atic/repo/litter_detect/videos/Blk 36-80m/DAY/Plastic of Refuse/converted2.mp4"  # Your video file

# Maximum display size
MAX_WIDTH = 1280
MAX_HEIGHT = 720

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.4)

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label)*12, y1), (0, 255, 0), -1)
            # Put label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Resize frame to fit screen while keeping aspect ratio
    height, width = frame.shape[:2]
    scale_w = MAX_WIDTH / width
    scale_h = MAX_HEIGHT / height
    scale = min(scale_w, scale_h, 1.0)  # Only downscale

    new_width = int(width * scale)
    new_height = int(height * scale)
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow("YOLO Detection", frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()