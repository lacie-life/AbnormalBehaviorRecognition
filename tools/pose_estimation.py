import cv2
import mediapipe as mp
import time
import torch


class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.boxes = None
        self.poses = None
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.upBody,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objects = self.detector(imgRGB).xyxy[0]

        humanObjects = []

        for obj in objects:
            if obj[5] == 0:
                humanObjects.append(obj)

        self.poses = torch.zeros((3, 33))
        self.boxes = torch.zeros((1, 4))

        tmpBB = []
        tmpPose = []

        if len(humanObjects) == 0:
            self.poses = torch.zeros((3, 33))
            self.boxes = torch.zeros((4, 1))
            return img, self.poses, self.boxes

        for obj in humanObjects:
            bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
            cropped_img = imgRGB[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            pose = self.pose.process(cropped_img)
            tmpBB.append(torch.tensor(bbox, dtype=torch.float32).reshape(4, 1))

            if pose.pose_landmarks:
                tmp_pose = []
                for id, lm in enumerate(pose.pose_landmarks.landmark):
                    h, w, c = cropped_img.shape
                    _cx, _cy = int(lm.x * w + int(obj[0])), int(lm.y * h + int(obj[1]))
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    tmp_pose.append([id, _cx, _cy])
                    if draw:
                        cv2.circle(cropped_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        imgRGB[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cropped_img
                tmpPose.append(torch.tensor(tmp_pose, dtype=torch.float32).transpose(0, 1))
            else:
                tmpPose.append(torch.zeros((3, 33)))

        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        if draw:
            for obj in humanObjects:
                bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        self.poses = torch.mean(torch.stack(tmpPose), dim=0)
        self.boxes = torch.mean(torch.stack(tmpBB), dim=0)

        return img, self.poses, self.boxes

    def getPosition(self, img, draw=True):
        lmList = []
        for pose in self.poses:
            tmp = []
            if pose.pose_landmarks:
                for id, lm in enumerate(pose.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    tmp.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                lmList.append(tmp)
        return lmList

    def findPoseMultiFrame(self, imgs):
        bboxes = []
        poses = []
        for img in imgs:
            _, bbox, pose = self.findPose(self, img)
            bboxes.append(bbox)
            poses.append(pose)
        return bboxes, poses


def main():
    cap = cv2.VideoCapture('/home/lacie/Datasets/KISA/train/Loitering/C001201_004.mp4')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-6.avi', fourcc, 20.0, size)

    pTime = 0

    detector = PoseDetector()

    while True:
        success, img = cap.read()

        if not success:
            break

        img, poses, boxes = detector.findPose(img)
        print(poses)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        out.write(img)
        cv2.waitKey(1)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
    
