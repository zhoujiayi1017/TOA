import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import argparse
import numpy as np

from posenet.pose_util import *
from posenet.posenet_pytorch_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--posenet_model_path', type=str, default='../models/pytorch/posenet.pth', help='顔検出用のpthモデルのパス')
parser.add_argument('--anchor_path', type=str, default='models/anchors.npy', help='顔検出用のアンカーのパス')
args = parser.parse_args()

posenet_model_path = args.posenet_model_path
anchor_path = args.anchor_path


posenet = PoseNet()
posenet.load_weights(posenet_model_path)
hight = posenet.hight
width = posenet.width

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
    # 画像の読み込み
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    frame = frame[:, int((w-h)/2):int((w-h)/2)+h] if h<w else frame[int((h-w)/2):int((h-w)/2)+w, :]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        count = 0

    # 予測
    input_image = cv2.resize(image, (hight, width))
    heatmaps, offsets = posenet.predict_on_image(input_image)

    # 予測結果の整理
    simple_kps = parse_output_2shoulder(heatmaps, offsets, 0.3)
    x, y, hightSize = get_faceCenterAndSize(simple_kps, hight)
    if (x and y and hightSize) is not None:
        faceROI = xyAndSize2ROI(x, y, hightSize, image.shape, margin_rate=1.1)
        cv2.rectangle(frame, (faceROI[0], faceROI[2]),(faceROI[1], faceROI[3]), (0,255,0), 3)

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv2.imshow(window_name, frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)