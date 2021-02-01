import numpy as np
import cv2

import tensorflow as tf

from detector import blazeface
from facemesh import FaceMesh

# 入力する動画と出力パスを指定。
target = "images/obama.mp4"
result = "images/test_output.m4v"

def make_grid(image, box, ratio=1.2):
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

blazeface_model_path = 'models/face_detection.tflite'
FaceMesh_model_path = "models/face_landmark.tflite"
anchor_path = 'anchors.npy'
count = 0
max_count = 10
fps = 0

detector = blazeface(blazeface_model_path, anchor_path)
mesh_maker = FaceMesh(FaceMesh_model_path)


movie = cv2.VideoCapture(target)
fps    = movie.get(cv2.CAP_PROP_FPS)
height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
width  = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter(result, int(fourcc), fps, (int(width), int(height)))
window_name = 'frame'

if not movie.isOpened():
    import sys
    sys.exit()

tm = cv2.TickMeter()
tm.start()

while True:
    ret, frame = movie.read()

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

    out.write(frame)
    cv2.imshow(window_name, frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)