import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpifpaf


class KeyPoints:

    def __init__(self):
        self.predictor = None

    def model(self, checkpoint="shufflenetv2k16"):
        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint)

    def detectPoints(self, frame, box):
        crop = frame[box[1]:box[3], box[0]:box[2]]

        frameRGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        predictions, gt_anns, meta = self.predictor.numpy_image(frameRGB)

        if predictions == []:
            predict = []
        else:
            predict = predictions[0].data[:, :2]
            predict[:, 0] += box[0]
            predict[:, 1] += box[1]

        return predict

