import xml.etree.ElementTree as ET
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
from datetime import timedelta


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
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

        self.model.cuda()
        self.model.eval()
        self.abnormal_detections = {}
        self.abnormal_time = {}
        self.event_start_time = None
        self.event_end_time = None
        self.loitering_offset = 0

    def detect_human(self, frame):

        objects = self.model(frame).xyxy[0]

        humanObjects = []

        tmpBB = []

        for obj in objects:
            if obj[5] == 0:
                humanObjects.append(obj)

        for obj in humanObjects:
            bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
            tmpBB.append(bbox)

        return tmpBB

    def _is_in_abnormal_area(self, box, abnormal_area):
        area = np.array(abnormal_area)

        top_left = (box[0], box[1])
        top_right = (box[2], box[1])
        bottom_left = (box[0], box[3])
        bottom_right = (box[2], box[3])

        if cv2.pointPolygonTest(area, top_left, False) < 0:
            return False
        if cv2.pointPolygonTest(area, top_right, False) < 0:
            return False
        if cv2.pointPolygonTest(area, bottom_left, False) < 0:
            return False
        if cv2.pointPolygonTest(area, bottom_right, False) < 0:
            return False

        return True

    def detect_abnormality(self, frame, frame_count):

        human_boxes = self.detect_human(frame)
        current_time = frame_count

        current_boxes = {tuple(box) for box in human_boxes}

        # draw abnormal area
        area = np.array(self.data_infor['abnormal_area'])
        cv2.polylines(frame, [area], True, (0, 0, 255), 2)

        if len(current_boxes) == 0 and self.event_start_time is not None:
            self.event_end_time = current_time

        # check bounding box in abnormal area
        event_append = False
        for box in current_boxes:
            if self._is_in_abnormal_area(box, self.data_infor['abnormal_area']):
                event_append = True
                # draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if event_append:
            if self.data_infor['abnormal_type'] == 'Intrusion':

                if self.event_start_time is None:
                    self.event_start_time = current_time

                if self.event_start_time is not None:
                    cv2.putText(frame, "Intrusion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if self.data_infor['abnormal_type'] == 'Loitering':
                # draw abnormal area
                area = np.array(self.data_infor['abnormal_area'])
                cv2.polylines(frame, [area], True, (0, 0, 255), 2)

                if self.loitering_offset is None:
                    self.loitering_offset = current_time
                if self.event_start_time is None and self.loitering_offset is not None and current_time - self.loitering_offset > 300:
                    self.event_start_time = current_time

                if self.event_start_time is not None:
                    cv2.putText(frame, "Loitering", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def process_video(self):
        print(self.data_infor["video_path"])
        cap = cv2.VideoCapture(self.data_infor["video_path"])
        video_name = self.data_infor["video_path"].split("/")[-1]
        video_name = video_name.replace(".mp4", "")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(f'{video_name}_output.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_abnormality(frame, frame_count)
            result.write(frame)

            cv2.waitKey(1)

            if frame_count % 100 == 0:
                print(frame_count)
            frame_count += 1

        cap.release()
        result.release()

        # convert to video time
        self.event_start_time = timedelta(seconds=self.event_start_time / 30)
        self.event_end_time = timedelta(seconds=self.event_end_time / 30)

        print("Video summary:")
        print(f"Video path: {self.data_infor['video_path']}")
        print(f"Type of event: {self.data_infor['abnormal_type']}")
        print(f"Start time of events: {self.event_start_time}")
        print(f"End time of events: {self.event_end_time}")


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
        video_name = self.data_infor["video_path"].split("/")[-1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        out = cv2.VideoWriter(f'{video_name}_output.mp4', fourcc, 20.0, (640, 480))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_human(frame)
            out.write(frame)
        #     cv2.imshow('Video', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        cap.release()
        # cv2.destroyAllWindows()