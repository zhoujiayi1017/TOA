import numpy as np
import cv2

import tensorflow as tf

class FaceMesh(object):
    def __init__(self, model_path):
        self.num_coords = 468
        self.x_scale = 192
        self.y_scale = 192

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict_on_image(self, img):
        # 画像の整形
        raw_shape = img.shape[:2]
        image = cv2.resize(img, (self.x_scale, self.y_scale)).reshape(tuple(self.input_shape)).astype(np.float32)/127.5-1.0
        # 推論
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        regression_out = self.interpreter.get_tensor(self.output_details[0]['index'])
        confidences_out = self.interpreter.get_tensor(self.output_details[1]['index'])
        # 推論結果の整理
        regression_out = regression_out.reshape(-1,3)
        regression_out[:,0] *= raw_shape[1]/self.x_scale
        regression_out[:,1] *= raw_shape[0]/self.y_scale

        return regression_out, confidences_out.reshape(-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    raw_image = cv2.imread("images/face.jpg")
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    model_path = "models/face_landmark.tflite"
    mesh_maker = FaceMesh(model_path)

    result, confidences = mesh_maker.predict_on_image(image)
    for xyz in result:
        cv2.circle(image, (xyz[0], xyz[1]), 3, (0, 255, 0), thickness=-1)
    plt.imshow(image)
    plt.show()
