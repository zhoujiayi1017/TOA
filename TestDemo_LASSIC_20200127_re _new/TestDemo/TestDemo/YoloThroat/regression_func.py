import numpy as np
import cv2
import glob
import pickle
from PIL import Image, ImageDraw
import os
from timeit import default_timer as timer

import pyrealsense2 as rs
from my_yolo import my_YOLO

def regression_input(persons):
    X=list()
    axis_x=1
    axis_y=1
    for person in persons:
        x=list()
        for f in person:
            x += f[2:]
        if len(x)==20:
            center_x = (x[0]+x[2])/2
            center_y = (x[1]+x[3])/2
            for i, value in enumerate(x):
                if i%2==0:
                    x[i]=(value-center_x)/axis_x
                else:
                    x[i]=(value-center_y)/axis_y
            x.append(center_x)
            x.append(center_y)
            X.append(x)
    if len(X)==0:
        return None
    else:
        return X

def regression_predict(inputs, model):
    inputs = np.array(inputs)
    inputs, opt = np.split(inputs,[20],axis=1)
    preds = model.predict(inputs)
    for i,o in enumerate(opt):
        preds[i, ::2] += o[0]
        preds[i,1::2] += o[1]
    return preds.tolist()

def draw_rectangle(image, preds):
    draw = ImageDraw.Draw(image)
    for pred in preds:
        draw.rectangle(pred, outline=(0, 255, 0))
    del draw
    return image
