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
import torch
from torchvision import models
import torch.nn as nn
import torchvision

classes = ['walk', 'fall', 'fight']

PATH = '/home/lacie/Github/AbnormalBehaviorRecognition/jellyfishABD/model_16_m3_0.8888.pth'

class KeyPoints:

    def __init__(self, model_path=''):
        self.predictor = self.model()
        self.classifier = models.resnext50_32x4d()
        self.classifier.fc = nn.Sequential(
            nn.Linear(self.classifier.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )
        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)


    def model(self, checkpoint="shufflenetv2k16"):
        predictor = openpifpaf.Predictor(checkpoint=checkpoint)
        return predictor

    def detectPoints(self, frame, box):
        crop = frame[box[1]:box[3], box[0]:box[2]]

        frameRGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        predictions, gt_anns, meta = self.predictor.numpy_image(frameRGB)

        pose_type = 'unknown'
        pose_type = self.check_pose_type(predictions, crop, box)

        if predictions == []:
            predict = []
        else:
            predict = predictions[0].data[:, :2]
            predict[:, 0] += box[0]
            predict[:, 1] += box[1]

        return predict, pose_type

    def check_pose_type(self, keypoints, crop, box):

        if self.fallDetection(keypoints, box):
            return 'fall'
        elif self.fight_detection(keypoints):
            return 'fight'
        else:
            return 'walk'

        # rect = crop.copy()
        # crop = cv2.resize(crop, (224, 224))
        # crop = torch.from_numpy(crop).float().to(self.device)  # Convert to tensor
        # crop = crop.unsqueeze(0)
        # crop = crop.permute(0, 3, 1, 2)
        #
        # predicted = self.classifier(crop)
        # cls_index = predicted.argmax(dim=1)
        # cls_prob = nn.functional.softmax(predicted, dim=1)
        #
        # pred_prob = cls_prob[0][cls_index].item()
        # pred_class = classes[cls_index]
        #
        # print(predicted, pred_class, pred_prob)
        # # pred_class = pred_class.cpu().numpy()[0]
        # print("Pose type: " + str(pred_class))
        #
        # print(rect.shape)
        #
        # if pred_class == 0:
        #     return 'walk'
        # elif pred_class == 2:
        #     return 'fall'
        # elif pred_class == 1:
        #     return 'fight'
        # elif rect.shape[0] < rect.shape[1]:
        #     return 'fall'
        # else:
        #     return 'unknown'

    def fallDetection(self, predict, box):
        # OpenPifPaf keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
        # left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

        if predict == []:
            keypoints = []
            return False
        else:
            keypoints = predict[0].data[:, :2]

        xmin, ymin = box[0], box[1]
        xmax, ymax = box[2], box[3]

        # Extract keypoints
        left_shoulder_y = keypoints[5][1]
        left_shoulder_x = keypoints[5][0]
        right_shoulder_y = keypoints[6][1]
        right_shoulder_x = keypoints[6][0]

        left_body_y = keypoints[11][1]
        left_body_x = keypoints[11][0]
        right_body_y = keypoints[12][1]
        left_foot_y = keypoints[15][1]
        right_foot_y = keypoints[16][1]

        len_factor = np.sqrt((left_shoulder_y - right_shoulder_y)**2 + (left_shoulder_x - right_shoulder_x)**2)
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx

        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
            len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
            return True
        return False

    def drawPoints(self, frame, points):
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        return frame

    def fall_detection2(self, keypoints):
        # OpenPifPaf keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
        # left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

        if keypoints == [] or len(keypoints) < 11:
            return False

        # Extract keypoints
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        # Calculate angles
        left_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # If the angles are less than a threshold (say, 30 degrees), the person might have fallen
        if left_leg_angle < 30 or right_leg_angle < 30:
            return True

        return False

    def fight_detection(self, keypoints):
        # OpenPifPaf keypoints: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
        # left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

        if keypoints == [] or len(keypoints) < 11:
            return False

        # Extract keypoints
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        # Calculate distances
        left_hand_head_dist = np.linalg.norm(left_wrist - nose)
        right_hand_head_dist = np.linalg.norm(right_wrist - nose)
        left_hand_body_dist = np.linalg.norm(left_wrist - left_shoulder)
        right_hand_body_dist = np.linalg.norm(right_wrist - right_shoulder)

        # Calculate angles
        left_angle = self.calculate_angle(left_wrist, nose, left_shoulder)
        right_angle = self.calculate_angle(right_wrist, nose, right_shoulder)

        # If the distances and angles are less than a threshold (say, 30 units and 30 degrees), the person might be fighting
        if (left_hand_head_dist < 30 or right_hand_head_dist < 30) and (
                left_hand_body_dist < 30 or right_hand_body_dist < 30) and (left_angle < 30 or right_angle < 30):
            return True

        return False

    def calculate_angle(self, a, b, c):
        # Calculates the angle between points a, b, and c
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

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
    keypoint = KeyPoints(model_path="/home/lacie/Github/AbnormalBehaviorRecognition/jellyfishABD/pose_resnext_400_epochs.pth")

    video_folder_path = "/home/lacie/Datasets/KISA/train/Violence/test"
    label = "fight"

    video_paths = [os.path.join(video_folder_path, file) for file in os.listdir(video_folder_path) if file.endswith(".mp4")]
    model = YOLO('/home/lacie/Github/AbnormalBehaviorRecognition/pre-train/yolov8x.pt')

    out_folder_path = "/home/lacie/Github/AbnormalBehaviorRecognition/final/data_pose/fight/"

    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    count = 0
    print(video_paths)
    for video_path in video_paths:

        print(video_path)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                bb = humanDetection(model, frame)
                if bb[0] != []:
                    for box in bb[0]:
                        points, pose_type = keypoint.detectPoints(frame, box)
                        if len(points) > 0:
                            frame = keypoint.drawPoints(frame, points)
                        cv2.putText(frame, pose_type, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                break
            frame_count += 1
            print(frame_count)
            cv2.putText(frame, f'Frame: {str(frame_count)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

