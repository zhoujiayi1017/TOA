#
# Run YOLO by realsense.
#
import cv2
import glob
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw
from timeit import default_timer as timer

import json
import win32pipe, win32file

import pyrealsense2 as rs
from my_yolo import my_YOLO
from regression_func import *

def on_predicted(preds, cpipe):
    #
    # 送信
    #
    #result_dict = {'x':str(preds[0]), 'y':'8'}
    result_dict = {'x':str(preds[0]), 'y':str(preds[1])}
    json_result = json.dumps(result_dict)
    win32file.WriteFile(cpipe, str(len(json_result)).encode('utf-8').zfill(4))
    win32file.WriteFile(cpipe, str(json_result).encode('utf-8'))
    #for pred in preds:
     #   result_dict = {'x':'12.34', 'y':'56.78'}
      #  json_result = json.dumps(result_dict)
       # win32file.WriteFile(cpipe, str(len(json_result)).encode('utf-8').zfill(4))
        #win32file.WriteFile(cpipe, str(json_result).encode('utf-8'))

def send_zero(cpipe):
    #
    # 送信
    #
    result_dict = {'x':'5', 'y':'10'}
    json_result = json.dumps(result_dict)
    win32file.WriteFile(cpipe, str(len(json_result)).encode('utf-8').zfill(4))
    win32file.WriteFile(cpipe, str(json_result).encode('utf-8'))


def calc_throat_pt(preds, depth_frame, depth_intrin, mirror_pt):
    """
    1. RealSenseのカメラ位置から喉位置までのx,y,z座標を求める
    2. ガルバノスキャナの光軸原点から喉位置までのx,y,z座標を求める

    Args:
        preds (list): 回帰された喉の領域を示す座標2点がリストで格納されている
                      x_左上、y_左上、x_右下、y_右下
        depth_frame (): realsense関連
        depth_intrin (): realsense関連
        mirror_pt (list): RealSenseに対するミラー制御位置の3次元座標 [x, y, z] 単位は[m]

    Returns:
        pts_xyz_calibrated (list): x/z,y/z,z座標をリストにして格納
    """
    pts_xyz_calibrated = []
    for pred in preds:
        pt_xy = [int((pred[0]+pred[2])/2), int((pred[1]+pred[3])/2)]
        tgt_depth = depth_frame.get_distance(pt_xy[0], pt_xy[1])
        tgt_pt_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [pt_xy[0], pt_xy[1]], tgt_depth)
        x, y, z = tgt_pt_xyz[0]-mirror_pt[0], tgt_pt_xyz[1]-mirror_pt[1], tgt_pt_xyz[2]-mirror_pt[2]
        if z == 0:
            z = 1
        x, y = x/z, y/z
        x, y, z = round(x, 3), round(y, 3), round(z, 3)
        xyz = list([x, y, z])
        pts_xyz_calibrated.append(xyz)
    return pts_xyz_calibrated

def overlap_rect(rect1, rect2):
    img1 = np.zeros((480,640), np.uint8)
    img2 = img1.copy()
    cv2.rectangle(img1, (int(rect1[0]),int(rect1[1])), (int(rect1[2]),int(rect1[3])), 1, -1)
    cv2.rectangle(img2, (int(rect2[0]),int(rect2[1])), (int(rect2[2]),int(rect2[3])), 1, -1)
    img_overlap = img1*img2
    return np.any(img_overlap==1)

def smoothing(preds, preds_stocked):
    """喉の位置の移動平均する関数

    Args:
        preds (list): 回帰された喉の領域を示す座標2点がリストで格納されている
                      x_左上、y_左上、x_右下、y_右下
        preds_stocked (list): 過去フレーム分の回帰された喉の領域を示す座標2点がリストで格納されている

    Returns:
        preds_tmp (list): 移動平均後の喉の領域を示す座標2点
                          x_左上、y_左上、x_右下、y_右下
        preds_stocked_tmp (list): 現フレームを含めた喉の領域を示す座標2点のリスト
    """
    max_stocked_num = 5
    preds_stocked_tmp = []
    preds_tmp = []

    if preds_stocked == []:
        for pred in preds:
            preds_stocked_tmp.append([pred])
            preds_tmp.append(pred)
        return preds_tmp, preds_stocked_tmp
    
    for pred in preds:
        if len(preds_stocked) == 0:
            preds_stocked_tmp.append([pred])
            preds_tmp.append(pred)
            
        for match_num, pred_stocked in enumerate(preds_stocked):
            pred_avg = list( np.mean(pred_stocked, axis=0) )
            overlap = overlap_rect(pred, pred_avg)
            if overlap:
                if len(pred_stocked)==max_stocked_num:
                    del pred_stocked[0]
                pred_stocked.append(pred)
                preds_stocked_tmp.append(pred_stocked)
                pred_avg = list( np.mean(pred_stocked, axis=0) )
                preds_tmp.append(pred_avg)
                del preds_stocked[match_num]
                break
            if match_num == len(preds_stocked)-1:
                preds_stocked_tmp.append([pred])
                preds_tmp.append(pred)
    return preds_tmp, preds_stocked_tmp

def put_text_fps(color_image, fps):
    """画像にfpsを描画する関数
    Args:
        color_image (numpy): 描画したい画像
        fps (str): fpsのテキスト
    """
    cv2.putText(color_image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.50, color=(0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
    cv2.putText(color_image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.50, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

def put_text_xyz(img, tgt_pts_xyz):
    """画像に喉位置の座標を描画する関数
    Args:
        img (numpy): 描画したい画像
        tgt_pts_xyz (list): 喉の位置[x/z, y/z, z]
    """
    for i, tgt_pt_xyz in enumerate(tgt_pts_xyz):
        x, y, z = tgt_pt_xyz[0], tgt_pt_xyz[1], tgt_pt_xyz[2]
        x, y = x*z, y*z
        x, y = round(x, 5), round(y, 5)
        text = str(i)+'. '+'x: {} [m] y: {} [m] z: {} [m]'.format(x, y, z)
        cv2.putText(img, text=text, org=(3, 35+20*i), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.50, color=(0,0,0), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img, text=text, org=(3, 35+20*i), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.50, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)

def run_detect():
    # ストリーム(Color/Depth)の設定
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーミング開始
    rspipe = rs.pipeline()
    profile = rspipe.start(config)

    # Alignオブジェクト生成     # 追加
    align_to = rs.stream.color  # 追加
    align = rs.align(align_to)  # 追加

    yolo = my_YOLO()
    Regression_model = pickle.load(open("C:/Users/ZHOU JIAYI/Desktop/realsense/TestDemo_LASSIC_20200127_re/TestDemo/TestDemo/YoloThroat/LR_model.sav", 'rb'))

    cpipe = win32pipe.CreateNamedPipe("\\\\.\\pipe\\pipe-throat",
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1, 65536, 65536, 300, None)

    print("接続待機中...")
    win32pipe.ConnectNamedPipe(cpipe, None)
    print("クライアント接続完了")

    prev_time = timer()
    fps = "FPS: ??"
    accum_time = 0
    curr_fps = 0

    # realsenseのカメラ原点からガルバノスキャナの光軸原点までの距離
	# 非常重要！！！realsense照相机开始到激光原点的距离
    mirror_pt = [-0.0395, -0.0413, -0.0508]
    # 移動平均
    preds_stocked = []

    try:
        while True:
            # フレーム待ち(Color & Depth)
            frames = rspipe.wait_for_frames()
            # color_frame = frames.get_color_frame()            # 削除
            # depth_frame = frames.get_depth_frame()            # 削除
            aligned_frames = align.process(frames)              # 追加
            color_frame = aligned_frames.get_color_frame()      # 追加
            depth_frame = aligned_frames.get_depth_frame()      # 追加
            if not depth_frame or not color_frame:
                continue
            # intrinsics                                                                # 追加
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 追加
            color_image = np.asanyarray(color_frame.get_data())
            # Depth画像
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            # 物体検出
            image = Image.fromarray(np.uint8(color_image))
            infos = yolo.detect_info(image)
            persons = yolo.person_detect(infos)
            persons = yolo.arrange_persons_info(persons)
            inputs = regression_input(persons)

            # 喉回帰
            throat_pts_xyz = []
            if inputs is not None:
                preds = regression_predict(inputs,Regression_model)
                preds, preds_stocked = smoothing(preds, preds_stocked)
                image = draw_rectangle(image,preds)
                throat_pts_xyz = calc_throat_pt(preds, depth_frame=depth_frame, depth_intrin=depth_intrin, mirror_pt=mirror_pt)

            if inputs is None:
                send_zero(cpipe)

            # 検出した顔のパーツを矩形で描画する
            #r_image = yolo.image_box(image,persons)            
            #color_image= np.array(r_image)
            # 検出した顔のパーツを矩形で描画しない
            color_image = np.array(image)

            if throat_pts_xyz:
                # C#へ送信
                on_predicted(throat_pts_xyz[0], cpipe)
                # 画像にx,y,zの座標を描画
                put_text_xyz(color_image, throat_pts_xyz)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            # 画像にfpsを描画
            put_text_fps(color_image, fps)


            # 表示
            # realsenseによる深度画像を連結して表示
            #images = np.hstack((color_image, depth_color_image))
            # realsenseによる深度画像を表示しない
            images = color_image
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        # ストリーミング停止
        rspipe.stop()
        cpipe.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detect()
