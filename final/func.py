import xml.etree.ElementTree as ET
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np


class XMLInfo:
    def __init__(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        self.numberArea = int(self.root.find(".//DetectionAreas").text)

        self.areas = []
        self.abnormal_area = []
        self.abnormal_type = None
        self.video_path = None

        if self.root.find(".//Filename") is not None:
            path = xml_path.replace(".xml", ".mp4")
            self.video_path = path

        if self.numberArea == 1:
            area = self.root.find(".//DetectArea")
            self.areas = [tuple(map(int, point.text.split(','))) for point in area.findall("Point")]
        if self.numberArea == 2:
            area = self.root.find(".//DetectArea")
            self.areas = [tuple(map(int, point.text.split(','))) for point in area.findall("Point")]
            if self.root.find(".//Intrusion") is not None:
                abnormal_area = self.root.find(".//Intrusion")
                self.abnormal_area = [tuple(map(int, point.text.split(','))) for point in
                                      abnormal_area.findall("Point")]
                self.abnormal_type = 'Intrusion'
            if self.root.find(".//Loitering") is not None:
                abnormal_area = self.root.find(".//Loitering")
                self.abnormal_area = [tuple(map(int, point.text.split(','))) for point in
                                      abnormal_area.findall("Point")]
                self.abnormal_type = 'Loitering'

    def get_root(self):
        return self.root

    def get_tree(self):
        return self.tree

    def get_video_infor(self):
        data = {}
        data["video_path"] = self.video_path
        data["areas"] = self.areas
        data["abnormal_area"] = self.abnormal_area
        data["abnormal_type"] = self.abnormal_type
        return data


class ILDetector:
    def __init__(self, data_infor):
        self.data_infor = data_infor
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.abnormal_detections = {}

    def detect_human(self, frame):
        frame = F.to_tensor(frame)
        frame = frame.unsqueeze(0)
        predictions = self.model(frame)
        labels = predictions[0]['labels'].numpy()
        boxes = predictions[0]['boxes'].detach().numpy()
        scores = predictions[0]['scores'].detach().numpy()
        human_boxes = boxes[labels == 1]
        human_scores = scores[labels == 1]
        human_boxes = human_boxes[human_scores > 0.5]
        return human_boxes

    def _is_in_abnormal_area(self, box, abnormal_area):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return cv2.pointPolygonTest(np.array(abnormal_area), (center_x, center_y), False) >= 0

    def detect_abnormality(self, frame):
        human_boxes = self.detect_human(frame)
        current_time = time.time()

        current_boxes = {tuple(box.tolist()) for box in human_boxes}

        for box_tuple in list(self.abnormal_detections.keys()):
            if box_tuple not in current_boxes and self._is_in_abnormal_area(np.array(box_tuple),
                                                                            self.data_infor['abnormal_area']):
                event_duration = current_time - self.abnormal_detections[box_tuple]
                print(f"Event for person {box_tuple} lasted {event_duration} seconds")
                del self.abnormal_detections[box_tuple]

        for box in human_boxes:
            if self._is_in_abnormal_area(box, self.data_infor['abnormal_area']):
                box_tuple = tuple(box.tolist())

                if self.data_infor['abnormal_type'] == 'Intrusion':
                    print("Intrusion detected")
                    self.abnormal_detections[box_tuple] = current_time
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(frame, 'Intrusion', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),
                                2)

                elif self.data_infor['abnormal_type'] == 'Loitering':
                    if box_tuple in self.abnormal_detections:
                        if current_time - self.abnormal_detections[box_tuple] > 10:
                            print("Loitering detected")
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, 'Loitering', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 2)
                    else:
                        self.abnormal_detections[box_tuple] = current_time
        return frame

    def process_video(self):
        cap = cv2.VideoCapture(self.data_infor["video_path"])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_abnormality(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


class SimpleABDetector:
    def __init__(self, data_infor):
        self.data_infor = data_infor
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.abnormal_detections = {}

    def detect_human(self, frame):
        frame = F.to_tensor(frame)
        frame = frame.unsqueeze(0)
        predictions = self.model(frame)
        labels = predictions[0]['labels'].numpy()
        boxes = predictions[0]['boxes'].detach().numpy()
        scores = predictions[0]['scores'].detach().numpy()
        human_boxes = boxes[labels == 1]
        human_scores = scores[labels == 1]
        human_boxes = human_boxes[human_scores > 0.5]

        # Draw bounding boxes
        for box in human_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        return frame

    def process_video(self):
        cap = cv2.VideoCapture(self.data_infor["video_path"])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_human(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
