import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'], gpu=True)
video_path = 'movies\Blk 36-80m\DAY\Plastic of Refuse\Video\plastic.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_width = 1280
output_height = int(frame_height * (output_width / frame_width)) 
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
frame_skip_yolo = 3
frame_skip_ocr = 30
frame_count = 0
latest_ocr_results = []
roi_x_start, roi_y_start = 0, 0   
roi_x_end, roi_y_end = 1280, 800 
yolo_model = YOLO(r"C:\dev\litter_segmentation\runs\detect\train46\weights\best.pt")
CONFIDENCE_THRESHOLD = 0.4
number_y_coords = []

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_text_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    _, black_text_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    combined_mask = cv2.bitwise_or(white_text_mask, black_text_mask)
    kernel = np.ones((2,2), np.uint8)
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    preprocessed_image = cv2.bitwise_not(opened_mask)
    return preprocessed_image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    processing_frame = cv2.resize(frame, (output_width, output_height))
    if frame_count % frame_skip_ocr == 0:
        text_roi = processing_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        # processing_frame = frame
        ocr_ready_frame = preprocess_for_ocr(text_roi)
        # cv2.rectangle(processing_frame,)

        allow_list = '0123456789-:'
        latest_ocr_results = reader.readtext(ocr_ready_frame, allowlist=allow_list)
        temp_coords = []
        for(bbox, text, prob) in latest_ocr_results:
            if prob >= CONFIDENCE_THRESHOLD:
                try:
                    number = int(text)
                    if 1 <= number <= 60:
                        y_center = (bbox[0][1] + bbox[2][1]) / 2
                        temp_coords.append({'number': number, 'y': int(y_center)})
                except ValueError:
                    continue
        number_y_coords = sorted(temp_coords, key=lambda item: item['y'])
    if frame_count % frame_skip_yolo == 0:
        yolo_results = yolo_model.track(processing_frame, persist=True)
        annotated_frame = yolo_results[0].plot()

        if yolo_results[0].boxes is not None and len(number_y_coords) > 1:
            trash_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            for trash_box in trash_boxes:
                trash_y_center = (trash_box[1] + trash_box[3]) /2
                location_text = "Unknown"
                for i in range(len(number_y_coords) - 1, 0, -1):
                    upper_num = number_y_coords[i]
                    lower_num = number_y_coords[i-1]
                    if lower_num['y'] < trash_y_center <= upper_num['y']:
                        location_text = f"Between{lower_num['number']} and {upper_num['number']}"
                        break
                cv2.putText(annotated_frame, location_text, (int(trash_box[0]), int(trash_box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        for coord in number_y_coords:
            cv2.line(annotated_frame, (0, coord['y']), (output_width, coord['y']), (0, 255, 255), 1)
            cv2.putText(annotated_frame, str(coord['number']), (10, coord['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        resized_frame = cv2.resize(annotated_frame, (output_width, output_height))
        cv2.imshow('Detections', resized_frame)
        out.write(resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()

    