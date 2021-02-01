import numpy as np
import cv2
import pyrealsense2 as rs

import tensorflow as tf

from detector import blazeface
from facemesh import FaceMesh

def make_grid(image, box, ratio=1.5):
    h, w = image.shape[:2]
    #具体的な座標を求める
    x_min, y_min = int(box[0]*h), int(box[1]*w)
    x_max, y_max = int(box[2]*h), int(box[3]*w)
    # 幅の大きい方を選択
    x, y = x_max-x_min, y_max-y_min
    base = x if x >= y else y
    # 1.2倍して両端のあまりを求める
    x_diff = int((base*1.2-x)/2)
    y_diff = int((base*1.2-x)/2)
    # 4辺をdiffだけ拡張した座標を返す
    x_min = x_min-x_diff if (x_min-x_diff) > 0 else 0
    y_min = y_min-y_diff if (y_min-y_diff) > 0 else 0
    x_max = x_max+x_diff if (x_max+x_diff) < w else w
    y_max = y_max+y_diff if (y_max+y_diff) < h else h
    return (x_min, y_min, x_max, y_max)

# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Alignオブジェクト作成（画角統一）
align_to = rs.stream.color
align = rs.align(align_to)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

window_name = 'RealSense'

blazeface_model_path = 'models/face_detection_back.tflite'
FaceMesh_model_path = "models/face_landmark.tflite"
anchor_path = 'anchors.npy'

detector = blazeface(blazeface_model_path, anchor_path)
mesh_maker = FaceMesh(FaceMesh_model_path)

while True:
    # フレーム待ち(Color & Depth)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame or not color_frame:
        continue
    #RGB画像
    color_image = np.asanyarray(color_frame.get_data())
    # Depth画像
    depth_color_frame = rs.colorizer().colorize(depth_frame)
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    frame = color_image
    h, w = frame.shape[:2]
    frame = frame[:, int((w-h)/2):int((w-h)/2)+h] if h<w else frame[int((h-w)/2):int((h-w)/2)+w, :]

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = detector.predict_on_image(image)
    h, w = frame.shape[:2]
    for box in preds:
        x_min, y_min, x_max, y_max = make_grid(image, box)
        patch = image[x_min:x_max, y_min:y_max].copy()
        result, _ = mesh_maker.predict_on_image(patch)
        for xyz in result:
            cv2.circle(frame[x_min:x_max, y_min:y_max], (xyz[0], xyz[1]), 1, (0, 255, 0), thickness=-1)
        #cv2.rectangle(frame, (y_min, x_min),(y_max, x_max), (0,255,0), 3)

    images = np.hstack((frame, depth_color_image))
    #images = cv2.addWeighted(color_image, alpha, depth_color_image, 1-alpha, 1.0)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)