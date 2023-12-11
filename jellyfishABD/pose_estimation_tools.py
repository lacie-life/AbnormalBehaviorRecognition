import time
import cv2
import numpy as np
import openpifpaf
from ultralytics import YOLO
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import joblib

class KeyPoints:

    def __init__(self):
        self.predictor = self.model()
        self.classifier = joblib.load('svm_weight.pkl')

    def model(self, checkpoint="shufflenetv2k16"):
        predictor = openpifpaf.Predictor(checkpoint=checkpoint)

        return predictor

    def detectPoints(self, frame, box):
        crop = frame[box[1]:box[3], box[0]:box[2]]

        frameRGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        predictions, gt_anns, meta = self.predictor.numpy_image(frameRGB)

        pose_type = 'unknown'

        if predictions == []:
            predict = []
        else:
            predict = predictions[0].data[:, :2]
            pose_type = self.check_pose_type(predict)
            predict[:, 0] += box[0]
            predict[:, 1] += box[1]

        return predict, pose_type

    def check_pose_type(self, keypoints):
        # OpenPifPaf keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
        # left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

        # # Extract keypoints
        # nose = keypoints[0]
        # left_wrist = keypoints[9]
        # right_wrist = keypoints[10]
        # left_ankle = keypoints[15]
        # right_ankle = keypoints[16]
        
        pred_class = self.classifier.predict(keypoints.reshape(1, -1))[0]
        if pred_class == 0: 
            return 'walk'
        elif pred_class == 1: 
            return 'fall'
        elif pred_class == 2: 
            return 'fight'

        # # Check for 'standing'
        # if nose[1] < min(left_ankle[1], right_ankle[1]):
        #     return 'standing'
        # # Check for 'fall down'
        # elif abs(nose[1] - min(left_ankle[1], right_ankle[1])) < 10:
        #     return 'fall down'
        # # Check for 'fighting'
        # elif max(left_wrist[1], right_wrist[1]) > nose[1]:
        #     return 'fighting'
        # else:
        #     return 'unknown'

    def drawPoints(self, frame, points):
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        return frame

def humanDetection(model, image):
    # objects = self.model(frame).xyxy[0]
    objects = model.predict(frame, classes=[0])

    humanObjects = objects[0].boxes.data

    # print(humanObjects)

    tmpBB = []
    conf = []
    tmp = []

    for obj in humanObjects:
        if obj[5] == 0:
            tmp.append(obj)

    for obj in tmp:
        bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
        tmpBB.append(bbox)

    return tmpBB, conf

if __name__ == "__main__":
    keypoint = KeyPoints()

    video_folder_path = "/home/lacie/Datasets/KISA/ver-4/walking"
    label = "walk"

    video_paths = [os.path.join(video_folder_path, file) for file in os.listdir(video_folder_path) if file.endswith(".mp4")]
    model = YOLO('/home/lacie/Github/AbnormalBehaviorRecognition/final/yolov8x.pt')

    for video_path in video_paths:

        cap = cv2.VideoCapture(video_path)

        video_name = video_path.split('/')[-1].split('.')[0]

        # write keypoint txt to file
        with open(f"/home/lacie/Github/AbnormalBehaviorRecognition/final/data_pose/{video_name}.txt", "w") as f:
            f.write(label + "\n")
            while True:
                ret, frame = cap.read()
                if ret:
                    boxes, conf = humanDetection(model, frame)

                    for box in boxes:

                        keypoints, pose_type = keypoint.detectPoints(frame, box)
                        frame = keypoint.drawPoints(frame, keypoints)

                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
                        cv2.putText(frame, pose_type, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if len(keypoints) > 0:
                            # f.write(str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " ")
                            # f.write("\n")
                            for i in range(len(keypoints)):
                                cv2.putText(frame, str(i), (int(keypoints[i][0]), int(keypoints[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                f.write(str(keypoints[i][0]) + " " + str(keypoints[i][1]) + " ")
                                f.write("\n")

                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()
