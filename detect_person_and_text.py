import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt



reader = easyocr.Reader(['en'], gpu=True)

# Open the video file (replace with your video file path)
video_path = 'movies\Blk 36-80m\DAY\Plastic of Refuse\Video\plastic.mp4'
cap = cv2.VideoCapture(video_path)

# --- 変更点 1: 元の動画から正確なサイズとFPSを取得 ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- 変更点 2: 出力動画のサイズを決定 ---
# ここでは表示しやすいように幅を1280pxにリサイズして保存する
output_width = 1280
output_height = int(frame_height * (output_width / frame_width)) # アスペクト比を維持

# Create a VideoWriter object (optional, if you want to save the output)
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

frame_skip_yolo = 3
frame_skip_ocr = 30
frame_count = 0
latest_ocr_results = []
roi_x_start, roi_y_start = 0, 0   
roi_x_end, roi_y_end = 1280, 800 

yolo_model = YOLO(r"C:\dev\litter_segmentation\runs\segment\train3\weights\best.pt")

CONFIDENCE_THRESHOLD = 0.4
number_y_coords = []

def preprocess_for_ocr(image):
   
    # 1. グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 明るい文字と暗い文字の分離・抽出
    # a) 明るい文字を抽出するためのマスクを作成 (閾値は要調整)
    # 輝度値が180より大きい部分を白(255)に、それ以外を黒(0)にする
    _, white_text_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

    # b) 暗い文字を抽出するためのマスクを作成 (閾値は要調整)
    # 輝度値が100より小さい部分を白(255)に、それ以外を黒(0)にする
    _, black_text_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # 3. 抽出した画像を合成 (OR演算)
    # 両方のマスクで白になっている部分を統合する
    combined_mask = cv2.bitwise_or(white_text_mask, black_text_mask)

    kernel = np.ones((2,2), np.uint8)
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # 4. 色の反転 (白背景に黒文字へ)
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

        for result in latest_ocr_results:
            coordinates = result[0]
            text = result[1]        # Get the text here!
            confidence = result[2]

            if confidence >= CONFIDENCE_THRESHOLD:
                original_top_left_x = int(coordinates[0][0]) + roi_x_start
                original_top_left_y = int(coordinates[0][1]) + roi_y_start
                
                original_bottom_right_x = int(coordinates[2][0]) + roi_x_start
                original_bottom_right_y = int(coordinates[2][1]) + roi_y_start
                
                cv2.rectangle(processing_frame, 
                            (original_top_left_x, original_top_left_y), 
                            (original_bottom_right_x, original_bottom_right_y), 
                            (0, 255, 0), 2)
                cv2.putText(
                    img=processing_frame,
                    text=f"Text: {text} ({confidence:.2f})",
                    org=(original_top_left_x, original_top_left_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=2
                )
        annotated_frame = yolo_results[0].plot()
        resized_frame = cv2.resize(annotated_frame, (output_width, output_height))
        cv2.imshow('Detections', resized_frame)
    
        out.write(resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()

    