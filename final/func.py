import xml.etree.ElementTree as ET
import time
import torch
import cv2
import numpy as np
from datetime import timedelta
from pose_estimation_tools import KeyPoints
from fire_tools import FireDetection
from bag_tools import AbandonmentDetector
from simple_tracker import SimpleTracker
from ultralytics import YOLO

from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.DATASETS.TRAIN = ("person_bag_train",)
    cfg.DATASETS.TEST = ("person_bag_val",)
    cfg.freeze()
    return cfg

class XMLInfo:
    def __init__(self, xml_path, output_folder_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.output_path = output_folder_path

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
        data["output"] = self.output_path
        return data


class ILDetector:
    def __init__(self, data_infor):
        self.data_infor = data_infor
        self.model = YOLO('/home/lacie/Github/AbnormalBehaviorRecognition/final/yolov8x.pt')
        self.abnormal_detections = {}
        self.abnormal_time = {}
        self.event_start_time = None
        self.event_end_time = None
        self.loitering_offset = 0

    def detect_human(self, frame):
        # objects = self.model(frame).xyxy[0]
        objects = self.model.predict(frame, classes=[0])

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
        print(self.data_infor['output'] + f'/{video_name}_output.avi')
        result = cv2.VideoWriter(self.data_infor['output'] + f'/{video_name}_output.avi',
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
        self.model = YOLO('/home/lacie/Github/AbnormalBehaviorRecognition/final/yolov8x.pt')
        self.abnormal_detections = {}
        self.fire_model = FireDetection(model_path="/home/lacie/Github/AbnormalBehaviorRecognition/final/fire-flame.pt")
        self.pose_model = KeyPoints()
        # self.bag_model = AbandonmentDetector()
        self.event_start_time = None
        self.event_end_time = None
        self.event_type = None
        self.draw = True
        self.tracker = SimpleTracker()
        self.pre_event = None

    def detect_human(self, frame):
        # objects = self.model(frame).xyxy[0]
        objects = self.model.predict(frame, classes=[0])

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

    def detect_fire(self, frame):
        return self.fire_model.detect(frame)

    def calculate_diff_frame(self, set_frames, frame):

        start_frame = set_frames[0]

        # subtract 2 image
        diff = cv2.absdiff(start_frame, frame)

        return diff

    def detect_aband(self, set_frames, frame):

        return True

    def check_fire(self, frame):
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        gray = cv2.GaussianBlur(gray, (100, 100), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    def process_video(self):

        cap = cv2.VideoCapture(self.data_infor["video_path"])
        video_name = self.data_infor["video_path"].split("/")[-1]
        video_name = video_name.replace(".mp4", "")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        print(self.data_infor['output'] + f'/{video_name}_output.avi')
        out = cv2.VideoWriter(self.data_infor['output'] + f'/{video_name}_output.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              10, size)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        if video_fps != 30.0:
            return "Video is not 30 FPS"

        frame_index = 0  # Frame Index
        step_size = (video_fps // video_fps)

        results = None

        previous_data = {
            'frame': [],
            'human_boxes': [],
            'human_poses': [],
            'pose_type': [],
            'objects': [],
            'obj_diff': []
        }

        human_boxes = []
        previous_obj = None
        objects = None
        obj_diff = 0.0

        start_detect = False

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            if frame_index % step_size == 0:

                oringinal_frame = frame.copy()

                human_boxes, conf = self.detect_human(frame)

                if len(human_boxes) > 0 and start_detect is False:
                    start_detect = True

                # Detect fire
                # fire_frame, isFire, prob = self.detect_fire(frame)

                # if self.event_start_time is None and self.event_type is None:
                #     if isFire == '1' or isFire == '2':
                #         self.event_start_time = frame_index
                #         self.event_type = 'Fire Detected'

                if start_detect:

                    diff = self.calculate_diff_frame(previous_data['frame'], frame)

                    # Check Abandonment
                    if self.event_start_time is None and self.event_type is None:
                        check_aban = self.detect_aband(previous_data['frame'], frame)


                    human_poses = []
                    pose_types = []
                    if len(human_boxes) > 0:
                        for box in human_boxes:
                            pose, tp = self.pose_model.detectPoints(frame, box)
                            human_poses.append(pose)
                            pose_types.append(tp)
                            print(tp)

                        # Update tracker
                        objects = self.tracker.update(human_boxes)
                        # Calculate distance between previous and current objects
                        obj_diff = 0.0

                        if previous_obj is None:
                            previous_obj = objects.items()
                        else:
                            for (obj, centroid) in objects.items():
                                for (prev_obj, prev_centroid) in previous_obj:
                                    if obj == prev_obj:
                                        obj_diff += np.linalg.norm(centroid - prev_centroid)
                        previous_obj = objects.items()

                    if len(previous_data['frame']) == 30:
                        previous_data['frame'] = previous_data['frame'][1:]
                        previous_data['human_boxes'] = previous_data['human_boxes'][1:]
                        previous_data['human_poses'] = previous_data['human_poses'][1:]
                        previous_data['pose_type'] = previous_data['pose_type'][1:]
                        previous_data['objects'] = previous_data['objects'][1:]
                        previous_data['obj_diff'] = previous_data['obj_diff'][1:]

                    # Check violence and fall down

                    # Check by diff
                    total_diff = 0.0
                    box_check = False
                    for box in range(len(previous_data['human_boxes'])):
                        if len(previous_data['human_boxes'][box]) > 0:
                            total_diff += previous_data['obj_diff'][box]

                    print(total_diff)
                    if self.event_start_time is None and self.event_type is None:
                        if total_diff > 100:
                            self.event_start_time = frame_index
                            self.event_type = 'Violence Detected'
                        if 10 > total_diff > 0:
                            self.event_start_time = frame_index
                            self.event_type = 'Fall Down Detected'

                    # Check by pose
                    total_pose = []
                    for pose in range(len(previous_data['pose_type'])):
                        for tp in previous_data['pose_type'][pose]:
                            if tp == 'unknown':
                                continue
                            total_pose.append(tp)

                    f_count = total_pose.count('fighting')
                    s_count = total_pose.count('standing')
                    l_count = total_pose.count('fall down')

                    if self.event_start_time is None and self.event_type is None:
                        if f_count > 30:
                            self.event_start_time = frame_index
                            self.event_type = 'Violence Detected'
                        elif l_count > 30:
                            self.event_start_time = frame_index
                            self.event_type = 'Fall Down Detected'

                    elif self.event_start_time is not None:
                        if self.event_type == 'Violence Detected':
                            if f_count == 0:
                                self.event_end_time = frame_index
                                self.event_type = None
                        elif self.event_type == 'Fall Down Detected':
                            if s_count > 10:
                                self.event_end_time = frame_index
                                self.event_type = None

                    previous_data['frame'].append(frame)
                    previous_data['human_boxes'].append(human_boxes)
                    previous_data['human_poses'].append(human_poses)
                    previous_data['pose_type'].append(pose_types)
                    previous_data['objects'].append(objects)
                    previous_data['obj_diff'].append(obj_diff)

                    if frame_index % 100 == 0:
                        print(frame_index)
                    frame_index += 1

                    if self.draw:
                        cv2.putText(oringinal_frame, f"Frame: {frame_index}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if self.event_start_time is not None:
                            cv2.putText(oringinal_frame, f"Event: {self.event_type}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    2)
                        elif self.event_start_time is None:
                            cv2.putText(oringinal_frame, f"Event: Normal", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255),
                                        2)
                        for obj in human_boxes:
                            bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                            cv2.rectangle(oringinal_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                        # print(human_poses)
                        for pose in human_poses:
                            for point in pose:
                                print(point)
                                if point[0] < 0 or point[1] < 0:
                                    continue
                                cv2.circle(oringinal_frame, (int(point[1]), int(point[2])), 5, (255, 0, 0), cv2.FILLED)

                out.write(oringinal_frame)
                cv2.waitKey(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

