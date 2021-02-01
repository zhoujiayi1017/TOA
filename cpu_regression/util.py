from math import atan, pi, degrees
import numpy as np
import cv2

import pyrealsense2 as rs

def calc_atan(pt_xyz):
    x, y, z = pt_xyz[0], pt_xyz[1], pt_xyz[2]
    if z == 0:
        z = 1
    x_theta = degrees(atan(x/z))
    y_theta = degrees(atan(y/z))

    x_theta = round(x_theta, 1)
    y_theta = round(y_theta, 1)

    return list([x_theta, y_theta])

def calc_deg(preds, depth_frame, depth_intrin):
    degrees=[]
    for pred in preds:
        pt_xy = [int((pred[0]+pred[2])/2), int((pred[1]+pred[3])/2)]
        tgt_depth = depth_frame.get_distance(pt_xy[0], pt_xy[1])
        tgt_pt_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [pt_xy[0], pt_xy[1]], tgt_depth)
        degree = calc_atan(tgt_pt_xyz)
        degrees.append(degree)
    return degrees

def overlap_rect(rect1, rect2):
    img1 = np.zeros((480,640), np.uint8)
    img2 = img1.copy()
    cv2.rectangle(img1, (int(rect1[0]),int(rect1[1])), (int(rect1[2]),int(rect1[3])), 1, -1)
    cv2.rectangle(img2, (int(rect2[0]),int(rect2[1])), (int(rect2[2]),int(rect2[3])), 1, -1)
    img_overlap = img1*img2
    return np.any(img_overlap==1)

def smoothing(preds, preds_stocked):
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

def put_text_deg(img, degrees):
    for i, degree in enumerate(degrees):
        text = str(i)+'. '+'x_deg: {}  y_deg: {}'.format(degree[0], degree[1])
        cv2.putText(img, text=text, org=(3, 35+20*i), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.50, color=(0,0,0), thickness=10, lineType=cv2.LINE_AA)
        cv2.putText(img, text=text, org=(3, 35+20*i), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.50, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)

def draw_features(image, boxes, landmarks):
    for box in boxes:
        print(box)
        box = [int(i) for i in box]
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    for marks in landmarks:
        for i,j in zip(marks[:5],marks[5:]):
            image = cv2.circle(image, (int(i),int(j)), 3, color=(255, 0, ),thickness=2)

    return image

def formatData(boxes, landmarks):
    features = list()
    for box, marks in zip(boxes, landmarks):
        feature = list()
        feature.extend(marks)
        feature.extend(box[:-1])

        features.append(feature)

    return features