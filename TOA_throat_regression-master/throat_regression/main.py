import cv2
import argparse
import numpy as np

import tensorflow as tf

from detector import blazeface
from facemesh import FaceMesh

parser = argparse.ArgumentParser()
parser.add_argument('--blazeface_model_path', type=str, default='models/face_detection.tflite', help='顔検出用のtfliteモデルのパス')
parser.add_argument('--FaceMesh_model_path', type=str, default='models/face_landmark.tflite', help='ランドマーク検出用のtfliteモデルのパス')
parser.add_argument('--anchor_path', type=str, default='models/anchors.npy', help='顔検出用のアンカーのパス')
args = parser.parse_args()

blazeface_model_path = args.blazeface_model_path
FaceMesh_model_path = args.FaceMesh_model_path
anchor_path = args.anchor_path

def make_grid(image, box, ratio=1.2):
    h, w = image.shape[:2]
    #具体的な座標を求める
    x_min, y_min = int(box[0]*h), int(box[1]*w)
    x_max, y_max = int(box[2]*h), int(box[3]*w)
    # 幅の大きい方を選択
    x, y = x_max-x_min, y_max-y_min
    base = x if x >= y else y
    # ratio倍して両端のあまりを求める
    x_diff = int((base*ratio-x)/2)
    y_diff = int((base*ratio-x)/2)
    # 4辺をdiffだけ拡張した座標を返す
    x_min = x_min-x_diff if (x_min-x_diff) > 0 else 0
    y_min = y_min-y_diff if (y_min-y_diff) > 0 else 0
    x_max = x_max+x_diff if (x_max+x_diff) < w else w
    y_max = y_max+y_diff if (y_max+y_diff) < h else h
    return (x_min, y_min, x_max, y_max)


detector = blazeface(blazeface_model_path, anchor_path)
mesh_maker = FaceMesh(FaceMesh_model_path)

count = 0
max_count = 10
fps = 0

cap = cv2.VideoCapture(0)
window_name = 'frame'

if not cap.isOpened():
    import sys
    sys.exit()

tm = cv2.TickMeter()
tm.start()

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    frame = frame[:, int((w-h)/2):int((w-h)/2)+h] if h<w else frame[int((h-w)/2):int((h-w)/2)+w, :]

    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        count = 0

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = detector.predict_on_image(image)
    h, w = frame.shape[:2]
    for box in preds:
        x_min, y_min, x_max, y_max = make_grid(image, box)
        patch = image[x_min:x_max, y_min:y_max].copy()
        result, confidences = mesh_maker.predict_on_image(patch)
        for xyz in result:
            cv2.circle(frame[x_min:x_max, y_min:y_max], (xyz[0], xyz[1]), 1, (0, 255, 0), thickness=-1)
        cv2.rectangle(frame, (y_min, x_min),(y_max, x_max), (0,255,0), 3)

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv2.imshow(window_name, frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)