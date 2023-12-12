import xml.etree.ElementTree as ET
import time
import torch
import cv2
import numpy as np
from datetime import timedelta
from pose_estimation_tools import KeyPoints
from fire_tools import FireDetection
from simple_tracker import SimpleTracker
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

MAX_DIFF = 100000

class SimpleABDetector:
    def __init__(self, data_infor):
        self.data_infor = data_infor
        self.model = YOLO('/home/lacie/Github/AbnormalBehaviorRecognition/pre-train/yolov8x.pt')
        self.abnormal_detections = {}
        self.fire_model = FireDetection(model_path="/home/lacie/Github/AbnormalBehaviorRecognition/pre-train/fire-yolov8.pt")
        self.pose_model = KeyPoints(model_path="/home/lacie/Github/AbnormalBehaviorRecognition/pre-train/pose_resnext_100.pth")
        # self.bag_model = AbandonmentDetector()
        self.event_start_time = None
        self.event_end_time = None
        self.event_type = None
        self.event_type_vis = None
        self.draw = True
        self.tracker = SimpleTracker()
        self.pre_event = None
        self.window_size = 120
        self.humanState = False # Human is in the scene before event
        self.eventAppeared = False # Event is appeared
        self.tmpEvent = None
        self.preTmpEnvent = None
        self.tmpEventTime = 0

    def export_to_xml(self):
        root = ET.Element("root")
        video_name = self.data_infor["video_path"].split("/")[-1]
        video_name = video_name.replace(".mp4", "")
        ET.SubElement(root, "VideoPath").text = self.data_infor["video_path"]
        ET.SubElement(root, "Type").text = self.event_type
        ET.SubElement(root, "StartTime").text = str(self.event_start_time)
        ET.SubElement(root, "EndTime").text = str(self.event_end_time)
        tree = ET.ElementTree(root)
        tree.write(self.data_infor["output"] + f"/{video_name}_result.xml")

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
        return self.fire_model.detect2(frame)

    def calculate_diff_frame(self, set_frames, frame, metric='mse', flag='frame'):

        if len(set_frames) < self.window_size - 1 and flag != 'bg':
            return MAX_DIFF
        
        diff = None

        if flag == 'fire':
            # subtract 2 image
            diff = cv2.absdiff(set_frames, frame)
            diff = np.sum(diff ** 2)/float(frame.shape[0] * frame.shape[1])
            return diff

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

        mean_diff = 0.0
        for i in range(len(previous_data['frame'])):
            mean_diff += self.calculate_diff_frame(previous_data['frame'], frame)
        mean_diff /= len(previous_data['frame'])

        diff_background = self.calculate_diff_frame([background_image], frame, flag='bg')

        human = False
        for bb in previous_data['human_boxes']:
            if len(bb) > 0:
                human = True
                break

        # print(diff)
        # print(diff_background)
        # print(human)
        # if (not human) and (diff > background_score - background_score * 0.005) and (diff < background_score + background_score * 0.005):
        # if (not human) and (diff_background > diff*0.1) and (diff > mean_diff - mean_diff * 0.005) and (diff < mean_diff + mean_diff * 0.005):
        if (not human) and diff_background > 30:
            print("Abandonment Detected")
            print("Current: " + str(diff))
            print("Background: " + str(background_score))
            print("Background Diff: " + str(diff_background))
            return True

        return False

    def check_fire(self, previous_data, frame, background_score, background_image, started=False):

        diff = 0.0
        if len(previous_data['frame']) <= 90:
            diff = self.calculate_diff_frame(previous_data['frame'], frame)
        else:
            diff = self.calculate_diff_frame(previous_data['frame'][90], frame, metric='mse', flag='fire')

        mean_diff = 0.0
        for i in range(len(previous_data['frame'])):
            mean_diff += self.calculate_diff_frame(previous_data['frame'], frame, metric='mse')
        mean_diff /= len(previous_data['frame'])

        diff_background = self.calculate_diff_frame([background_image], frame, flag='bg')

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            
            print("=====================================================================================")
            print(diff)
            print(diff_background)
            # print(human)

            count = 0
            for indx in range(80, len(previous_data['frame'])):
                bb = self.detect_fire(previous_data['frame'][indx])
                # print(bb)
                if len(bb) > 0:
                    count += 1

            # if human and diff > 100:
            if human:
                # Re check fire
                # print(diff_background)
                print("Fire frame: " + str(count))
                if count > 30 and diff_background > 50:
                    print("Fire Detected")
                    print("Current: " + str(diff))
                    print("Background: " + str(background_score))
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

        mean_diff = 0.0
        for i in range(len(previous_data['frame'])):
            mean_diff += self.calculate_diff_frame(previous_data['frame'], frame)
        mean_diff /= len(previous_data['frame'])

        diff_background = self.calculate_diff_frame([background_image], frame, flag='bg')

        # Check by pose
        total_pose = []
        for pose in range(len(previous_data['pose_type'])):
            for tp in previous_data['pose_type'][pose]:
                if tp == 'unknown':
                    continue
                total_pose.append(tp)

        f_count = total_pose.count('fight')
        s_count = total_pose.count('walk')
        fa_count = total_pose.count('fall')

        print("===================================")
        print(fa_count)
        print(f_count)
        print(s_count)
        print(diff_background)
        print(mean_diff)
        print(background_score)

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)

            # if human and diff_background > 50 and fa_count > 30:
            if human and diff_background > 50 and mean_diff < background_score + background_score * 0.02 and mean_diff > background_score - background_score * 0.05 and fa_count > 10:
                print("Fall Detected")
                return 'start'
        else:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            # print(diff)
            if human and diff_background > 50 and s_count > 10:
                print("Fall Detected End")
                return 'end'

    def check_fight(self, previous_data, frame, background_score, background_image, started=False):

        diff = self.calculate_diff_frame(previous_data['frame'], frame)

        diff_background = self.calculate_diff_frame([background_image], frame, flag='bg')

        # Check by pose
        total_pose = []
        for pose in range(len(previous_data['pose_type'])):
            for tp in previous_data['pose_type'][pose]:
                if tp == 'unknown':
                    continue
                total_pose.append(tp)

        f_count = total_pose.count('fight')
        s_count = total_pose.count('walk')
        fa_count = total_pose.count('fall')

        print("===================================")
        print(fa_count)
        print(f_count)
        print(diff)
        print(background_score)

        if not started:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break

            # if human and diff > background_score + background_score * 0.2 and f_count > 30:
            if human and f_count > 30 and diff_background > 50: 
                print("Fight Detected")
                return 'start'
        else:
            human = False
            for bb in previous_data['human_boxes']:
                if len(bb) > 0:
                    human = True
                    break
            if human and s_count > 10 and f_count < 10:
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
            'idx': [],
            'frame': [],
            'human_boxes': [],
            'human_poses': [],
            'pose_type': [],
            'objects': [],
            'obj_diff': []
        }

        previous_obj = None
        objects = None
        obj_diff = 0.0

        start_detect = False

        conf = 0.6

        background_score = 0.0
        background_image = None
        tmp = 0

        warm_up = 0

        ret, firse_frame = cap.read()

        previous_data['frame'].append(firse_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            human_boxes = []
            human_poses = []
            pose_types = []

            if frame_index % step_size == 0:

                oringinal_frame = frame.copy()

                human_boxes, r_conf = self.detect_human(frame, conf=conf)

                diff = self.calculate_diff_frame(previous_data['frame'], frame, flag='bg')

                # Waiting for human appear
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

                
                if start_detect is True:

                    # Warm up
                    if frame_index < warm_up + (2*video_fps):
                        if len(previous_data['frame']) == self.window_size:
                            previous_data['idx'] = previous_data['idx'][1:]
                            previous_data['frame'] = previous_data['frame'][1:]
                            previous_data['human_boxes'] = previous_data['human_boxes'][1:]
                            previous_data['human_poses'] = previous_data['human_poses'][1:]
                            previous_data['pose_type'] = previous_data['pose_type'][1:]
                            previous_data['objects'] = previous_data['objects'][1:]
                            previous_data['obj_diff'] = previous_data['obj_diff'][1:]

                        previous_data['idx'].append(frame_index)
                        previous_data['frame'].append(frame)
                        previous_data['human_boxes'].append(human_boxes)
                        previous_data['human_poses'].append([])
                        previous_data['pose_type'].append([])
                        previous_data['objects'].append([])
                        previous_data['obj_diff'].append(0.0)
                        frame_index += 1
                        print("Warm up")
                        out.write(oringinal_frame)

                        continue

                    # Get human bounding box and pose
                    if len(human_boxes) > 0:
                        print("Human detected")
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
                    if self.event_start_time is None and self.event_type is None:
                        print("Check abandonment")
                        check_aband = self.detect_aband(previous_data, frame, background_score, background_image)
                        if check_aband: self.tmpEvent = 'aband'
                        if check_aband and self.tmpEventTime < 30 and self.tmpEvent == 'aband':
                            self.tmpEvent = 'aband'
                            self.tmpEventTime += 1
                            print(self.tmpEventTime)
                        elif check_aband and self.tmpEventTime >= 30:
                            self.event_start_time = frame_index
                            self.event_type = 'Abandonment'
                            self.event_type_vis = 'Abandonment Detected'
                            bag = diff = cv2.absdiff(background_image, frame)
                            self.tmpEvent = None
                            self.tmpEventTime = 0
                            print("Abandonment Detected: " + str(frame_index))
                            exit(0)

                    # Check fire
                    if self.event_start_time is None and self.event_type is None:
                        print("Check fire")
                        check_fire = self.check_fire(previous_data, frame, background_score, background_image)
                        if check_fire: self.tmpEvent = 'fire'
                        if check_fire == 'start'and self.tmpEventTime < 30 and self.tmpEvent == 'fire':
                            self.tmpEvent = 'fire'
                            self.tmpEventTime += 1
                            print(self.tmpEventTime)
                        elif check_fire == 'start' and self.tmpEventTime >= 30:
                            self.event_start_time = frame_index
                            self.event_type = 'FireDetection'
                            self.event_type_vis = 'Fire Detected'
                            self.tmpEvent = None
                            self.tmpEventTime = 0
                            print("Fire Detected: " + str(frame_index))
                            # exit(0)
                    elif self.event_start_time is not None and self.event_type == 'FireDetection':
                        check_fire = self.check_fire(previous_data, frame, background_score, background_image, started=True)
                        if not check_fire == 'end':
                            self.event_end_time = frame_index
                            self.event_type_vis = 'Normal'

                    # Check fall down
                    if self.event_start_time is None and self.event_type is None:
                        print("Check fall")
                        check_fall = self.check_fall(previous_data, frame, background_score, background_image)
                        if check_fall: self.tmpEvent = 'fall'
                        if check_fall == 'start'and self.tmpEventTime < 10 and self.tmpEvent == 'fall':
                            self.tmpEvent = 'fall'
                            self.tmpEventTime += 1
                            print(self.tmpEventTime)
                        elif check_fall == 'start' and self.tmpEventTime >= 10:
                            self.event_start_time = frame_index
                            self.event_type = 'Falldown'
                            self.event_type_vis = 'Falldown Detected'
                            self.tmpEvent = None
                            self.tmpEventTime = 0
                            print("Fall down Detected: " + str(frame_index))
                            exit(0)
                    elif self.event_start_time is not None and self.event_type == 'Falldown':
                        check_fall = self.check_fall(previous_data, frame, background_score, background_image, started=True)
                        if check_fall == 'end':
                            self.event_end_time = frame_index
                            self.event_type_vis = 'Normal'

                    # Check violence
                    if self.event_start_time is None and self.event_type is None:
                        print("Check fight")
                        check_fight = self.check_fight(previous_data, frame, background_score, background_image)
                        if check_fight: self.tmpEvent = 'fight'
                        if check_fight == 'start'and self.tmpEventTime < 30 and self.tmpEvent == 'fight':
                            self.tmpEvent = 'fight'
                            self.tmpEventTime += 1
                            print(self.tmpEventTime)
                        elif check_fight == 'start' and self.tmpEventTime >= 30:
                            self.event_start_time = frame_index
                            self.event_type = 'Violence'
                            self.event_type_vis = 'Fight Detected'
                            self.tmpEvent = None
                            self.tmpEventTime = 0
                            print("Fight Detected: " + str(frame_index))
                            exit(0)
                    elif self.event_start_time is not None and self.event_type == 'Violence':
                        check_fight = self.check_fight(previous_data, frame, background_score, background_image, started=True)
                        if not check_fight:
                            self.event_end_time = frame_index
                            self.event_type = 'Normal'

                    # Update previous data
                    if self.tmpEvent != self.preTmpEnvent:
                        print("Reset tmp event")
                        self.tmpEventTime = 0
                    
                    self.preTmpEnvent = self.tmpEvent

                    if len(previous_data['frame']) == self.window_size:
                        previous_data['idx'] = previous_data['idx'][1:]
                        previous_data['frame'] = previous_data['frame'][1:]
                        previous_data['human_boxes'] = previous_data['human_boxes'][1:]
                        previous_data['human_poses'] = previous_data['human_poses'][1:]
                        previous_data['pose_type'] = previous_data['pose_type'][1:]
                        previous_data['objects'] = previous_data['objects'][1:]
                        previous_data['obj_diff'] = previous_data['obj_diff'][1:]

                        print("Update previous data")
                        print("===================================")
                        print("idx: " + str(previous_data['idx'][-1]))
                        print("Number frame: " + str(len(previous_data['frame'])))
                        print("Number bounding box: " + str(len(previous_data['human_boxes'][-1])))
                        print("Number pose: " + str(len(previous_data['human_poses'][-1])))
                        print("\n")

                    previous_data['idx'].append(frame_index)
                    previous_data['frame'].append(frame)
                    previous_data['human_boxes'].append(human_boxes)
                    previous_data['human_poses'].append(human_poses)
                    previous_data['pose_type'].append(pose_types)
                    previous_data['objects'].append(objects)
                    previous_data['obj_diff'].append(obj_diff)

                    if self.draw:
                        if len(human_boxes) > 0:
                            i = 0
                            for obj in human_boxes:
                                bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                                cv2.rectangle(oringinal_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                                cv2.putText(oringinal_frame, pose_types[i], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                i += 1
                            # print(human_poses)
                            for keypoints in human_poses:
                                if len(keypoints) > 0:
                                    for i in range(len(keypoints)):
                                        cv2.putText(frame, str(i), (int(keypoints[i][0]), int(keypoints[i][1])),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if frame_index % 100 == 0:
                    print(frame_index)
                frame_index += 1

                cv2.putText(oringinal_frame, f"Frame: {frame_index}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.event_start_time is not None:
                    cv2.putText(oringinal_frame, f"Event: {self.event_type_vis}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                        2)
                elif self.event_start_time is None:
                    cv2.putText(oringinal_frame, f"Event: Normal", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255),
                                            2)

                cv2.imshow("Frame", oringinal_frame)
                out.write(oringinal_frame)
                cv2.waitKey(1)

        if self.event_end_time is None: 
            self.event_end_time = frame_index
        if self.event_start_time is None:
            self.event_start_time = 0
        if self.event_end_time is not None and self.event_start_time is not None:
            self.event_start_time = timedelta(seconds=self.event_start_time / 30)
            self.event_end_time = timedelta(seconds=self.event_end_time / 30)

        print("Video summary:")
        print(f"Video path: {self.data_infor['video_path']}")
        print(f"Type of event: {self.event_type}")
        print(f"Start time of events: {self.event_start_time}")
        print(f"End time of events: {self.event_end_time}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Export to xml file
        self.export_to_xml()


