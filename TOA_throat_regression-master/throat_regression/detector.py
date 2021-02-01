import numpy as np
import cv2

import tensorflow as tf

class blazeface(object):
    def __init__(self, model_path, anchor_path):
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 256
        self.y_scale = 256
        self.h_scale = 256
        self.w_scale = 256
        #最初にざっくりとbboxを削減する閾値
        self.min_score_thresh = 0.75
        #iouの計算時にどれだけ重なっているboxを考慮するか
        self.min_suppression_threshold = 0.3

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        self.anchors = np.load(anchor_path)
        self.sigmoid = lambda x : 1 / (1 + np.exp(-x))

    def predict_on_image(self, img):
        # 画像の整形
        image = cv2.resize(img, (self.x_scale, self.y_scale)).reshape(tuple(self.input_shape)).astype(np.float32)/127.5-1.0
        # 推論
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        regression_out = self.interpreter.get_tensor(self.output_details[0]['index'])
        classification_out = self.interpreter.get_tensor(self.output_details[1]['index'])
        # 推論結果の整理
        boxes = self._decode_boxes(regression_out)
        scores = self._decode_scores(classification_out)
        # 重み付け非最大値抑制
        preds = self._weighted_non_max_suppression(boxes, scores)

        return preds

    def _decode_boxes(self, regression_out):
        x_center = regression_out[...,0]/self.x_scale*self.anchors[:,2]+self.anchors[:,0]
        y_center = regression_out[...,1]/self.y_scale*self.anchors[:,3]+self.anchors[:,1]
        w = regression_out[..., 2] / self.w_scale * self.anchors[:, 2]
        h = regression_out[..., 3] / self.h_scale * self.anchors[:, 3]

        ymin = y_center - h / 2.  # ymin
        xmin = x_center - w / 2.  # xmin
        ymax = y_center + h / 2.  # ymax
        xmax = x_center + w / 2.  # xmax

        boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)

        return boxes[0]

    def _decode_scores(self, classification_out):
        x = classification_out[0,:,0].clip(-self.score_clipping_thresh, self.score_clipping_thresh)
        return self.sigmoid(x)

    def _weighted_non_max_suppression(self, boxes, cls_scores):
        output_detections = []
        idxes = np.where(cls_scores>self.min_score_thresh)[0]
        boxes = boxes[idxes]
        cls_scores = cls_scores[idxes]

        # 検出されたものを降順に並び替え
        remaining = np.argsort(cls_scores)[::-1]

        while len(remaining) > 1:
            first_box = boxes[remaining[0]]
            first_scores = cls_scores[remaining[0]]
            other_boxes = boxes[remaining]
            ious = overlap_similarity(first_box, other_boxes)

            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            weighted_boxes = first_box.copy()
            weighted_scores = first_scores.copy()
            if len(overlapping) > 1:
                coordinates = boxes[overlapping]
                scores = cls_scores[overlapping]
                total_score = scores.sum()
                weighted_boxes = (coordinates * np.expand_dims(scores,1)).sum(axis=0) / total_score
                weighted_scores = total_score/len(scores)

            output_detections.append(np.append(weighted_boxes, weighted_scores))

        return np.stack(output_detections) if len(output_detections) > 0 else np.zeros((0, 5))

def intersect(box_a, box_b, A, B):
    max_xy = np.fmin(np.expand_dims(box_a[:, 2:],axis=1).repeat(B,axis=1),
                    np.expand_dims(box_b[:, 2:],axis=0).repeat(A,axis=0))
    min_xy = np.fmax(np.expand_dims(box_a[:, :2],axis=1).repeat(B,axis=1),
                    np.expand_dims(box_b[:, :2],axis=0).repeat(A,axis=0))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    inter = intersect(box_a, box_b, A, B)
    area_a = np.expand_dims((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]),axis=1).repeat(B,axis=1)  # [A,B]
    area_b = np.expand_dims((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]),axis=0).repeat(A,axis=0)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def overlap_similarity(box, other_boxes):
    return np.squeeze(jaccard(np.expand_dims(box, axis=0), other_boxes))


if __name__ == '__main__':
    # デモ
    import matplotlib.pyplot as plt

    model_path = 'models/face_detection.tflite'
    anchor_path = 'models/anchors.npy'
    detector = blazeface(model_path, anchor_path)


    raw_image = cv2.imread("photo.jpg")
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    preds = detector.predict_on_image(raw_image)
    image = raw_image.copy()
    h, w = image.shape[:2]
    for box in preds:
        cv2.rectangle(image, (int(box[1]*w),int(box[0]*h)),(int(box[3]*w),int(box[2]*h)), (0,255,0), 3)
    plt.imshow(image)
    plt.show()
