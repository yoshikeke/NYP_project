import cv2
from ultralytics import YOLO
import easyocr
# 事前学習済みのセグメンテーションモデルをロード
model = YOLO("yolo11n.pt") 
# 動画ファイルへのパス
video_path = r"C:\dev\litter_segmentation\movies\Blk 36-80m\DAY\Plastic of Refuse\Video\plastic.mp4"
# model.track()を実行するが、自動表示はオフにする (show=False)
results = model.track(video_path, tracker="bytetrack.yaml", stream=True, classes=[0])
# ウィンドウの名前を定義
window_name = "YOLOv8 Tracking"

reader = easyocr.Reader(['en'])
results_ch = reader.readtext(video_path)

for r in results:
    # 検出結果をフレームに描画
    im_array = r.plot()  # r.plot()はNumPy配列 (フレーム) を返す
    
    # ------------------- ここが重要な部分 -------------------
    # フレームを指定した幅にリサイズする（例：幅を1280ピクセルに）
    # アスペクト比は維持する
    h, w, _ = im_array.shape
    target_width = 1280
    scale = target_width / w
    target_height = int(h * scale)
    
    # リサイズ実行
    resized_frame = cv2.resize(im_array, (target_width, target_height))
    # ----------------------------------------------------
    
    # リサイズしたフレームを表示
    cv2.imshow(window_name, resized_frame)
    
    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# すべてのウィンドウを閉じる
cv2.destroyAllWindows()