import xml.etree.ElementTree as ET
import cv2
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

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

        
        path = xml_path.replace(".map", ".mp4")
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
    def __init__(self, data_infor, pretrain_path=None, debug=False, visual=False):
        self.data_infor = data_infor
        self.model = YOLO( pretrain_path + '/yolov8x.pt')
        self.abnormal_detections = {}
        self.abnormal_time = {}
        self.event_type = 'Normal'
        self.event_start_time = None
        self.event_end_time = None
        self.loitering_offset = None
        self.debug = debug
        self.visual = visual
        self.event_state = False
        self.duration = None

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

    # TODO: Re-check the time
    def detect_abnormality(self, frame, frame_count, pre_frames):

        human_boxes_each_frame = []
        total_bb = 0
        for frame in pre_frames:
            bb, cof = self.detect_human(frame)
            human_boxes_each_frame.append(bb)
            if len(bb) > 0:
                total_bb = total_bb + 1
        
        current_time = frame_count

        human_boxes = self.detect_human(frame)

        # print(human_boxes)

        # draw abnormal area
        area = np.array(self.data_infor['abnormal_area'])
        cv2.polylines(frame, [area], True, (0, 0, 255), 2)

        if total_bb == 0 and self.event_start_time is not None and self.event_state is False:
            self.event_end_time = current_time
            self.event_type = 'Normal'
            self.event_state = True

        # check bounding box in abnormal area
        event_append = False
        if total_bb > 0:
            for bb_each in human_boxes_each_frame:
                if len(bb_each) > 0:
                    for box in bb_each:
                    # print(box)
                        if self._is_in_abnormal_area(box, self.data_infor['abnormal_area']):
                            event_append = True
                            # draw bounding box
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        if event_append:
            if self.data_infor['abnormal_type'] == 'Intrusion':

                # draw abnormal area
                area = np.array(self.data_infor['abnormal_area'])
                cv2.polylines(frame, [area], True, (0, 0, 255), 2)

                if self.event_start_time is None:
                    self.event_type = 'Intrusion'
                    self.event_start_time = current_time

                # if self.event_start_time is not None:
                #     cv2.putText(frame, "Intrusion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if self.data_infor['abnormal_type'] == 'Loitering':
                # draw abnormal area
                area = np.array(self.data_infor['abnormal_area'])
                cv2.polylines(frame, [area], True, (0, 0, 255), 2)

                if self.loitering_offset is None:
                    self.loitering_offset = current_time
                if self.event_start_time is None and self.loitering_offset is not None and current_time - self.loitering_offset > 400:
                    self.event_type = 'Loitering'
                    self.event_start_time = current_time

                # if self.event_start_time is not None:
                #     cv2.putText(frame, "Loitering", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame
    
    def export_to_xml(self):
        root = ET.Element("root")
        video_name = self.data_infor["video_path"].split("/")[-1]
        video_name = video_name.replace(".mp4", "")
        ET.SubElement(root, "VideoPath").text = self.data_infor["video_path"]
        ET.SubElement(root, "Type").text = self.data_infor["abnormal_type"]
        ET.SubElement(root, "StartTime").text = str(self.event_start_time)
        ET.SubElement(root, "EndTime").text = str(self.event_end_time)
        ET.SubElement(root, "Duration").text = str(self.duration)
        tree = ET.ElementTree(root)
        tree.write(self.data_infor["output"] + f"/{video_name}_result.xml")

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

        pre_frame = []

        pre_frame.append(cap.read()[1])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            origin_frame = frame.copy()
            frame = self.detect_abnormality(origin_frame, frame_count, pre_frame)
            
            result.write(frame)
            cv2.waitKey(1)

            if self.visual:
                cv2.putText(frame, "Frame: " + str(frame_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Event: " + self.event_type, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("frame", frame)
                cv2.waitKey(1)

            if frame_count % 100 == 0:
                print(frame_count)
            frame_count += 1

            if len(pre_frame) == 10:
                pre_frame = pre_frame[1:]

            pre_frame.append(origin_frame)

        cap.release()
        result.release()

        # convert to video time
        if self.event_start_time is None:
            self.event_start_time = 0
        if self.event_end_time is None:
            self.event_end_time = frame_count

        self.duration = self.event_end_time - self.event_start_time
        self.event_start_time = timedelta(seconds=self.event_start_time / 30)
        self.event_end_time = timedelta(seconds=self.event_end_time / 30)
        self.duration = timedelta(seconds=self.duration / 30)

        if self.debug:
            print("Video summary:")
            print(f"Video path: {self.data_infor['video_path']}")
            print(f"Type of event: {self.data_infor['abnormal_type']}")
            print(f"Start time of events: {self.event_start_time}")
            print(f"End time of events: {self.event_end_time}")
            print(f"Duration of events: {self.duration}")

        # Export to xml file
        self.export_to_xml()


