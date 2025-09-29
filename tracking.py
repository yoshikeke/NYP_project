import cv2
import numpy as np
import easyocr
import supervision as sv
from ultralytics import YOLO

# --- 1. SETUP ---

# Define file paths
VIDEO_SOURCE_PATH = r"C:\dev\litter_segmentation\movies\Blk 36-80m\DAY\Plastic of Refuse\Video\plastic.mp4"  # <-- IMPORTANT: UPDATE THIS
VIDEO_OUT_PATH = r"C:\dev\litter_segmentation\movies\Blk 36-80m\DAY\Plastic of Refuse\Video" # <-- IMPORTANT: UPDATE THIS

# Load the models (this is done only once)
print("Loading models...")
yolo_model = YOLO("yolov8n.pt")  # Using a standard pre-trained YOLO model
ocr_reader = easyocr.Reader(['en'])
print("Models loaded.")

box_annotator_person = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
label_annotator_person = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_scale=0.5,
    text_position=sv.Position.TOP_CENTER
)

# For Text (EasyOCR) - Green boxes
box_annotator_text = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
label_annotator_text = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_scale=0.5,
    text_position=sv.Position.TOP_CENTER
)
# --- 2. VIDEO PROCESSING ---

# Setup video capture and writer
cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))


# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. RUN INFERENCE ---

    # A. YOLO People Detection
    # We specify class 0, which is 'person' in the COCO dataset
    yolo_results = yolo_model(frame, classes=[0], conf=0.5)[0]
    detections_person = sv.Detections.from_ultralytics(yolo_results)
    
    # B. EasyOCR Text Detection
    ocr_results = ocr_reader.readtext(frame)

    # --- 4. ANNOTATE FRAME ---
    
    # Start with the original frame
    annotated_frame = frame.copy()

    # Annotate with YOLO person detections
    person_labels = [
        f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections_person
    ]
    # Use the person-specific annotators
    annotated_frame = box_annotator_person.annotate(
        scene=annotated_frame,
        detections=detections_person
    )
    annotated_frame = label_annotator_person.annotate(
        scene=annotated_frame,
        detections=detections_person,
        labels=person_labels
    )

    # Annotate with EasyOCR text detections
    if ocr_results:
        # (The code to prepare OCR results for Supervision remains the same...)
        xyxy_text, conf_text, label_text = [], [], []
        for bbox, text, confidence in ocr_results:
            x_min = int(min([point[0] for point in bbox]))
            y_min = int(min([point[1] for point in bbox]))
            x_max = int(max([point[0] for point in bbox]))
            y_max = int(max([point[1] for point in bbox]))
            xyxy_text.append([x_min, y_min, x_max, y_max])
            conf_text.append(confidence)
            label_text.append(text)

        detections_text = sv.Detections(
            xyxy=np.array(xyxy_text),
            confidence=np.array(conf_text)
        )
        
        # Use the text-specific annotators
        annotated_frame = box_annotator_text.annotate(
            scene=annotated_frame,
            detections=detections_text
        )
        annotated_frame = label_annotator_text.annotate(
            scene=annotated_frame,
            detections=detections_text,
            labels=label_text
        )

        # --- 5. WRITE AND DISPLAY ---
        
        out.write(annotated_frame)
        cv2.imshow("Annotated Frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- 6. CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete. Annotated video saved to: {VIDEO_OUT_PATH}")