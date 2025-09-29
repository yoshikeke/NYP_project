import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Load your YOLO model (replace with your model's path)
model = YOLO('yolo11n.pt', task='detect')

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

frame_skip = 2
frame_count = 0

def preprocess_for_ocr(image):
    """
    OCRに適した画像に前処理を行う関数。
    明るい文字と暗い文字を別々に抽出し、白背景・黒文字の画像に変換する。

    :param image: 入力画像 (OpenCV形式, BGR)
    :return: 前処理後の画像 (グレースケール)
    """
    # 1. グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 明るい文字と暗い文字の分離・抽出
    # a) 明るい文字を抽出するためのマスクを作成 (閾値は要調整)
    # 輝度値が180より大きい部分を白(255)に、それ以外を黒(0)にする
    _, white_text_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # b) 暗い文字を抽出するためのマスクを作成 (閾値は要調整)
    # 輝度値が100より小さい部分を白(255)に、それ以外を黒(0)にする
    _, black_text_mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV)

    # 3. 抽出した画像を合成 (OR演算)
    # 両方のマスクで白になっている部分を統合する
    combined_mask = cv2.bitwise_or(white_text_mask, black_text_mask)

    kernel = np.ones((4,4), np.uint8)
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # 4. 色の反転 (白背景に黒文字へ)
    preprocessed_image = cv2.bitwise_not(opened_mask)
    
    return preprocessed_image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue
    ocr_ready_frame = preprocess_for_ocr(frame)

    allow_list = '0123456789-:'
    text_detections = reader.readtext(ocr_ready_frame, allowlist=allow_list)

    for result in text_detections:
        coordinates = result[0]
        text = result[1]        # Get the text here!
        confidence = result[2]
        
        # --- FIX IS HERE ---
        # Get the top-left and bottom-right points from the coordinates list
        top_left = tuple(map(int, coordinates[0]))
        bottom_right = tuple(map(int, coordinates[2]))
        
        # Draw the bounding box using the corrected points
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
        # Draw the detected text on the frame (no need to run OCR again)
        cv2.putText(
            img=frame,
            text=f"Text: {text} ({confidence:.2f})",
            org=(top_left[0], top_left[1] - 10), # Position text above the box
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 255),
            thickness=2
        )
    resized_frame = cv2.resize(frame, (output_width, output_height))
    resized_ocr_frame = cv2.resize(ocr_ready_frame, (output_width, output_height))
    cv2.imshow('Detections', resized_frame)
    cv2.imshow("Preprocessed for OCR", resized_ocr_frame)
    out.write(resized_frame)
    out.write(resized_ocr_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1
cap.release()
out.release()
cv2.destroyAllWindows()

    