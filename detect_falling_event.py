import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import math

# --- 設定 ---
MODEL_PATH = "C:/dev/litter_segmentation/runs/segment/train3/weights/best.pt"
VIDEO_PATH = r"C:\dev\litter_segmentation\data\throwing\HRL -80m_Camera 01_20240325162243_20240325162801.mp4"
CONFIDENCE_THRESHOLD = 0.5

# --- トラッキングと落下判定のロジック ---
# 追跡しているオブジェクトの情報を保存する辞書
# キー: オブジェクトID, 値: 中止点の座標リスト
track_history = defaultdict(list)
# 次に割り当てるオブジェクトID
next_track_id = 0
# 落下判定用のカウンター
falling_counters = defaultdict(int)
# 落下と判定するまでに必要な連続フレーム数
FALLING_FRAMES_THRESHOLD = 3 
# 落下と判定するY座標の最低移動量
MIN_Y_MOVEMENT = 10

# 1. YOLOv8モデルをロード
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# 2. ビデオをロード
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    exit()

# 3. フレームごとにビデオを処理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # A. モデルで推論を実行
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    
    # B. 現在のフレームで検出されたオブジェクトの中心点を取得
    current_centroids = []
    if results[0].masks is not None:
        for mask in results[0].masks:
            contour = mask.xy[0].astype(np.int32)
            # モーメントを計算して中心点を見つける
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centroids.append((cx, cy))

    # C. 新しいオブジェクトと既存のオブジェクトをマッチング
    if not track_history: # 最初のフレームの場合
        for cx, cy in current_centroids:
            track_history[next_track_id].append((cx, cy))
            next_track_id += 1
    else:
        # 追跡中のオブジェクトIDと中心点のリストを作成
        tracked_ids = list(track_history.keys())
        last_known_points = [points[-1] for points in track_history.values()]
        
        # 使われていないオブジェクトIDと中心点を追跡
        unmatched_centroids = list(range(len(current_centroids)))
        
        # 既存のオブジェクトと新しい中心点を距離でマッチング
        for i, (track_id, last_point) in enumerate(zip(tracked_ids, last_known_points)):
            # 最も近い中心点を見つける
            min_dist = float('inf')
            best_match_idx = -1
            
            for j, centroid in enumerate(current_centroids):
                if j in unmatched_centroids:
                    dist = math.hypot(last_point[0] - centroid[0], last_point[1] - centroid[1])
                    if dist < min_dist and dist < 50: # 50ピクセル以内なら同じオブジェクトとみなす
                        min_dist = dist
                        best_match_idx = j
            
            if best_match_idx != -1:
                # マッチ成功
                cx, cy = current_centroids[best_match_idx]
                
                # 落下判定
                prev_cx, prev_cy = last_point
                if cy > prev_cy + MIN_Y_MOVEMENT: # Y座標が増加（下に移動）した場合
                    falling_counters[track_id] += 1
                else:
                    falling_counters[track_id] = 0 # 下に移動していないのでリセット
                
                track_history[track_id].append((cx, cy))
                unmatched_centroids.remove(best_match_idx)
        
        # マッチしなかった新しい中心点は、新しいオブジェクトとして登録
        for idx in unmatched_centroids:
            track_history[next_track_id].append(current_centroids[idx])
            next_track_id += 1

    # D. 結果を描画
    overlay = frame.copy()
    if results[0].masks is not None:
        for mask, box in zip(results[0].masks, results[0].boxes):
            contour = mask.xy[0].astype(np.int32)
            
            # このマスクの中心点に近い追跡IDを見つける
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 最も近いIDを探す
                min_dist = float('inf')
                current_id = -1
                for track_id, points in track_history.items():
                    dist = math.hypot(points[-1][0] - cx, points[-1][1] - cy)
                    if dist < min_dist:
                        min_dist = dist
                        current_id = track_id
                
                # 落下中かどうかで色を変える
                is_falling = falling_counters.get(current_id, 0) >= FALLING_FRAMES_THRESHOLD
                color = (0, 0, 255) if is_falling else (0, 255, 0) # 落下中は赤、それ以外は緑
                
                cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)
                if is_falling:
                    cv2.putText(overlay, "FALLING", (box.xyxy[0][0].int(), box.xyxy[0][1].int() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


    # 結果をブレンドして表示
    alpha = 0.4
    result_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Falling Event Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()