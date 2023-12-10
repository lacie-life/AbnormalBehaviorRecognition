import xml.etree.ElementTree as ET
import time
import torch
import cv2
import numpy as np
from datetime import timedelta
from pose_estimation_tools import KeyPoints
from fire_tools import FireDetection
# from bag_tools import AbandonmentDetector
from simple_tracker import SimpleTracker
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

MAX_DIFF = 100000

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
        objects = self.model.predict(frame, classes=[0], verbose=False)

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
        self.window_size = 60

    def detect_human(self, frame, conf=0.6):
        # objects = self.model(frame).xyxy[0]
        conf = float(conf)
        objects = self.model.predict(frame, classes=[0], conf=conf, verbose=False)

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

    def calculate_diff_frame(self, set_frames, frame, metric='ssim'):

        if len(set_frames) < self.window_size - 1:
            return MAX_DIFF
        
        diff = None

        if metric == 'mse':

            start_frame = set_frames[0]

            # subtract 2 image
            diff = cv2.absdiff(start_frame, frame)

            diff = np.sum(diff ** 2)/float(frame.shape[0] * frame.shape[1])
        elif metric == 'ssim':
            start_frame = set_frames[0]

            gray_image1 = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # # subtract 2 image
            # diff = cv2.compareSSIM(gray_image1, gray_image2, full=True)
            diff = ssim(gray_image1, gray_image2)

        return diff

    def detect_aband(self, previous_data, frame, background_score, background_image):

        diff = self.calculate_diff_frame(previous_data['frame'], frame)

        diff_background = self.calculate_diff_frame([background_image], frame)

        human = False
        for bb in previous_data['human_boxes']:
            if len(bb) > 0:
                human = True
                break

        print(diff)
        if (not human) and (diff > background_score - background_score * 0.005) and (diff < background_score + background_score * 0.005):
            print("Abandonment Detected")
            print("Current: " + str(diff))
            print("Background: " + str(background_score))
            print("Background Diff: " + str(diff_background))
            exit(0)
            return True

        return False

    def check_fire(self, previous_data, frame, background_score, background_image, started=False):

        diff = self.calculate_diff_frame(previous_data['frame'], frame)

        diff_background = self.calculate_diff_frame([background_image], frame)

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)
            if not human and diff > background_score + background_score * 0.02:
                # Re check fire
                count = 0
                for frame in previous_data['frame']:
                    fire_frame, isFire, prob = self.detect_fire(frame)
                    if (isFire == '1' or isFire == '2') and prob > 0.5:
                        count += 1

                if count > 10:
                    print("Fire Detected")
                    print("Current: " + str(diff))
                    print("Background: " + str(background_score))
                    exit(0)
                    return 'start'
        else:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            if not human and diff < background_score + background_score * 0.1:
                print("Fire Detected End")
                return 'end'

    def check_fall(self, previous_data, frame, background_score, background_image, started=False):

        diff = self.calculate_diff_frame(previous_data['frame'], frame)

        diff_background = self.calculate_diff_frame([background_image], frame)

        # Check by pose
        total_pose = []
        for pose in range(len(previous_data['pose_type'])):
            for tp in previous_data['pose_type'][pose]:
                if tp == 'unknown':
                    continue
                total_pose.append(tp)

        f_count = total_pose.count('fighting')
        s_count = total_pose.count('standing')
        fa_count = total_pose.count('fall down')

        print("===================================")
        print(fa_count)
        print(f_count)

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)

            if human and diff < background_score + background_score * 0.001 and fa_count > 30:
                print("Fall Detected")
                return 'start'
        else:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)
            if human and diff > background_score + background_score * 0.02 and s_count > 10:
                print("Fall Detected End")
                return 'end'

    def check_fight(self, previous_data, frame, background_score, started=False):

        diff = self.calculate_diff_frame(previous_data['frame'], frame)

        # Check by pose
        total_pose = []
        for pose in range(len(previous_data['pose_type'])):
            for tp in previous_data['pose_type'][pose]:
                if tp == 'unknown':
                    continue
                total_pose.append(tp)

        f_count = total_pose.count('fighting')
        s_count = total_pose.count('standing')
        fa_count = total_pose.count('fall down')

        print("===================================")
        print(fa_count)
        print(f_count)

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)

            if human and diff < background_score + background_score * 0.2 and f_count > 30:
                print("Fight Detected")
                return 'start'
        else:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)
            if human and diff > background_score + background_score * 0.5 and s_count > 10:
                print("Fight Detected End")
                return 'end'

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
        human_poses = []
        pose_types = []
        previous_obj = None
        objects = None
        obj_diff = 0.0

        start_detect = False

        conf = 0.6

        background_score = 0.0
        background_image = None
        tmp = 0

        warm_up = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            if frame_index % step_size == 0:

                oringinal_frame = frame.copy()

                human_boxes, r_conf = self.detect_human(frame, conf=conf)

                diff = self.calculate_diff_frame(previous_data['frame'], frame)

                if len(human_boxes) > 0 and start_detect is False:
                    start_detect = True
                    conf = 0.3
                    warm_up = frame_index

                if self.window_size * 1.5 < frame_index < 600 + self.window_size * 1.5:
                    background_score += diff
                    tmp += 1
                if frame_index == 600 + self.window_size * 1.5:
                    background_score /= tmp
                    background_image = frame.copy()

                if start_detect is True and frame_index > warm_up + (2*video_fps):

                    # Get human bounding box and pose
                    if len(human_boxes) > 0:
                        for box in human_boxes:
                            pose, tp = self.pose_model.detectPoints(frame, box)
                            human_poses.append(pose)
                            pose_types.append(tp)
                            # print(tp)
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

                    # TODO: Review all threshold and parameters
                    # TODO: Update pose classification based on skeleton model
                    # Check Abandonment

                    # Check fire
                    if self.event_start_time is None and self.event_type is None:
                        check_fire = self.check_fire(previous_data, frame, background_score, background_image)
                        if check_fire == 'start':
                            self.event_start_time = frame_index
                            self.event_type = 'Fire Detected'
                    elif self.event_start_time is not None and self.event_type == 'Fire Detected':
                        check_fire = self.check_fire(previous_data, frame, background_score, background_image, started=True)
                        if not check_fire == 'end':
                            self.event_end_time = frame_index
                            self.event_type = 'Normal'

                    # Check fall down
                    if self.event_start_time is None and self.event_type is None:
                        check_fall = self.check_fall(previous_data, frame, background_score, background_image)
                        if check_fall == 'start':
                            self.event_start_time = frame_index
                            self.event_type = 'Fall Detected'
                    elif self.event_start_time is not None and self.event_type == 'Fall Detected':
                        check_fall = self.check_fall(previous_data, frame, background_score, background_image, started=True)
                        if check_fall == 'end':
                            self.event_end_time = frame_index
                            self.event_type = 'Normal'

                    # Check violence
                    if self.event_start_time is None and self.event_type is None:
                        check_fight = self.check_fight(previous_data, frame, background_score)
                        if check_fight:
                            self.event_start_time = frame_index
                            self.event_type = 'Fight Detected'
                    elif self.event_start_time is not None and self.event_type == 'Fight Detected':
                        check_fight = self.check_fight(previous_data, frame, background_score, started=True)
                        if not check_fight:
                            self.event_end_time = frame_index
                            self.event_type = 'Normal'
                    
                    # Check abandonment
                    if self.event_start_time is None and self.event_type is None:
                        check_aband = self.detect_aband(previous_data, frame, background_score, background_image)
                        if check_aband:
                            self.event_start_time = frame_index
                            self.event_type = 'Abandonment Detected'

                if len(previous_data['frame']) == 60:
                    previous_data['frame'] = previous_data['frame'][1:]
                    previous_data['human_boxes'] = previous_data['human_boxes'][1:]
                    previous_data['human_poses'] = previous_data['human_poses'][1:]
                    previous_data['pose_type'] = previous_data['pose_type'][1:]
                    previous_data['objects'] = previous_data['objects'][1:]
                    previous_data['obj_diff'] = previous_data['obj_diff'][1:]

                previous_data['frame'].append(frame)
                previous_data['human_boxes'].append(human_boxes)
                previous_data['human_poses'].append(human_poses)
                previous_data['pose_type'].append(pose_types)
                previous_data['objects'].append(objects)
                previous_data['obj_diff'].append(obj_diff)

                human_boxes.clear()
                human_poses.clear()
                pose_types.clear()

                if self.draw:
                    cv2.putText(oringinal_frame, f"Frame: {frame_index}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if self.event_start_time is not None:
                        cv2.putText(oringinal_frame, f"Event: {self.event_type}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    2)
                    elif self.event_start_time is None:
                        cv2.putText(oringinal_frame, f"Event: Normal", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255),
                                        2)
                    if len(human_boxes) > 0:
                        for obj in human_boxes:
                            bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                            cv2.rectangle(oringinal_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                        # print(human_poses)
                        for keypoints in human_poses:
                            if len(keypoints) > 0:
                                for i in range(len(keypoints)):
                                    cv2.putText(frame, str(i), (int(keypoints[i][0]), int(keypoints[i][1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if frame_index % 100 == 0:
                    print(frame_index)
                frame_index += 1

                cv2.imshow("Frame", oringinal_frame)
                out.write(oringinal_frame)
                cv2.waitKey(1)

        print("Video summary:")
        print(f"Video path: {self.data_infor['video_path']}")
        print(f"Type of event: {self.event_type}")
        print(f"Start time of events: {self.event_start_time}")
        print(f"End time of events: {self.event_end_time}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

