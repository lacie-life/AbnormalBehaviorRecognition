import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
from tools.pose_estimation import PoseDetector


class simpleKISADataLoader(Dataset):
    def __init__(self, xml_file, video_folder, transform=None):
        self.xml_file = xml_file
        self.video_folder = video_folder
        self.transform = transform
        self.clips = self.parse_xml()
        self.pose_detector = PoseDetector()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_info = self.clips[idx]
        video_path = f"{self.video_folder}/{video_info['filename']}"

        video_frames = self.load_video_frames(video_path)

        poses, bounding_boxes = self.extract_human_information(video_frames)

        event_label = video_info['event']
        start_time = video_info['start_time']
        duration = video_info['duration']

        if self.transform:
            video_frames = [self.transform(frame) for frame in video_frames]

        return {
            'frames': video_frames,
            'bounding_boxes': bounding_boxes,
            'poses': poses,
            'label': event_label,
            'start_time': start_time,
            'duration': duration
        }

    def parse_xml(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        clips = []
        for clip in root.iter('Clip'):
            filename = clip.find('Header/Filename').text
            event = clip.find('Alarms/Alarm/AlarmDescription').text
            start_time = clip.find('Alarms/Alarm/StartTime').text
            duration = clip.find('Alarms/Alarm/AlarmDuration').text

            clips.append({
                'filename': filename,
                'event': event,
                'start_time': start_time,
                'duration': duration
            })

        return clips

    def load_video_frames(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def extract_human_information(self, frames):
        _, poses, bounding_boxes = self.pose_detector.findPose(frames)
        return poses, bounding_boxes
