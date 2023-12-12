from torchfusion_utils.models import load_model, save_model
import torch
from torchvision import datasets, transforms, models
import numpy as np
from torch.autograd import Variable
import cv2
from PIL import Image
from deepstack_sdk import ServerConfig, Detection
from ultralytics import YOLO

class FireDetection:
    def __init__(self, model_path):
        # self.model = torch.load(model_path)
        # self.model.eval()
        # self.model = self.model.cuda()
        self.model = YOLO(model_path)
        self.transformer = transforms.Compose([transforms.Resize(size=(224, 224)),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5],
                                                       [0.5, 0.5, 0.5])])

    def detect(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'))
        orig = img.copy()
        img_processed = self.transformer(img).unsqueeze(0)
        img_var = Variable(img_processed, requires_grad=False)
        img_var = img_var.cuda()
        logp = self.model(img_var)
        expp = torch.softmax(logp, dim=1)
        confidence, clas = expp.topk(1, dim=1)

        co = confidence.item() * 100

        class_no = str(clas).split(',')[0]
        class_no = class_no.split('(')
        class_no = class_no[1].rstrip(']]')
        class_no = class_no.lstrip('[[')

        orig = np.array(orig)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (800, 500))

        if class_no == '1':
            label = "Neutral: " + str(co) + "%"
            cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif class_no == '2':
            label = "Smoke: " + str(co) + "%"
            cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif class_no == '0':
            label = "Fire: " + str(co) + "%"
            cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return orig, class_no, co

    def detect2(self, img):
        objects = self.model.predict(img, classes=[0], verbose=False, conf=0.5)
        fireObjects = objects[0].boxes.data

        tmpBB = []

        for i in range(len(fireObjects)):
            tmpBB.append(fireObjects[i].tolist())
            # print(fireObjects[i].tolist())

        return tmpBB


if __name__ == "__main__":
    model_path = "/home/lacie/Github/AbnormalBehaviorRecognition/jellyfishABD/fire-yolov8.pt"
    fire_detection = FireDetection(model_path=model_path)

    video_path = "/home/lacie/Datasets/KISA/train/Abandonment/test/C096102_001.mp4"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        bb = fire_detection.detect2(img)

        for i in range(len(bb)):
            cv2.rectangle(img, (int(bb[i][0]), int(bb[i][1])), (int(bb[i][2]), int(bb[i][3])), (0, 255, 0), 2)
            cv2.putText(img, str(bb[i][4]), (int(bb[i][0]), int(bb[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("img", img)

        cv2.waitKey(1)
    cv2.destroyAllWindows()
