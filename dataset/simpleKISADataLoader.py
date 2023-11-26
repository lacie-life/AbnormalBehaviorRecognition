import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from datetime import datetime
import cv2
import os
import numpy as np
from tools.pose_estimation import PoseDetector

event_list = {'Abandonment': 0,
              'Falldown': 1,
              'FireDetection': 2,
              'Intrusion': 3,
              'Loitering': 4,
              'Violence': 5,
              'Normal': 6}


def time2second(time_string):
    time_format = "%H:%M:%S"
    time_object = datetime.strptime(time_string, time_format)
    seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
    return seconds


class simpleKISADataLoader(Dataset):
    def __init__(self, root_folder, sample=10, transform=None):
        self.root_folder = root_folder
        self.sample = sample

        self.event_folders = [folder for folder in os.listdir(root_folder) if
                              os.path.isdir(os.path.join(root_folder, folder))]
        self.file_list = []

        for event_folder in self.event_folders:
            event_path = os.path.join(root_folder, event_folder)
            videos = [file for file in os.listdir(event_path) if file.endswith('.mp4')]
            for video in videos:
                video_path = os.path.join(event_path, video)
                xml_file = video.replace('.mp4', '.xml')
                xml_path = os.path.join(event_path, xml_file)
                if os.path.exists(xml_path):
                    self.file_list.append((video_path, xml_path))

        self.transform = transform
        self.pose_detector = PoseDetector()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video_path, xml_path = self.file_list[idx]

        video_frames = self.load_video_frames(video_path)

        bboxes, poses = self.extract_human_information(video_frames)

        event_label, start_time, duration = self.parse_xml(xml_path)

        start_time = torch.tensor([start_time])
        duration = torch.tensor([duration])

        if self.transform:
            video_frames = [self.transform(frame) for frame in video_frames]

        # print("Number frames: " + str(len(video_frames)))
        # print("Number human objects: " + str(len(bboxes)))
        # print("Number human poses: " + str(len(poses)))
        # print(event_label)
        # print(start_time)
        # print(duration)

        return [
            video_frames,
            bboxes,
            poses,
            event_label,
            start_time,
            duration
        ]

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # print(xml_path)

        # Extract label information from XML
        event_label = root.find('.//AlarmDescription').text
        start_time = root.find(".//StartTime").text
        duration = root.find(".//AlarmDuration").text

        return event_list[event_label], time2second(start_time), time2second(duration)

    def load_video_frames(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_count % self.sample == 0:
                frames.append(frame)
            frame_count += 1
        return frames

    def extract_human_information(self, frames):
        bboxes = torch.zeros((len(frames), 4))
        poses = torch.zeros((len(frames), 3, 33))
        i = 0
        for frame in frames:
            _, pose, bbox = self.pose_detector.findPose(frame)
            bboxes[i] = bbox
            poses[i] = pose
            i += 1

        return bboxes, poses


def collate_fn(rets):
    frames = [ret[0] for ret in rets]
    bboxes = [ret[1] for ret in rets]
    poses = [ret[2] for ret in rets]
    event = [ret[3] for ret in rets]
    start_time = [ret[4] for ret in rets]
    duration = [ret[5] for ret in rets]

    bboxes = torch.stack(bboxes)
    poses = torch.stack(poses)

    res = (
        torch.tensor(frames),
        torch.tensor(bboxes),
        torch.tensor(poses),
        torch.tensor(event),
        torch.tensor(start_time),
        torch.tensor(duration)
    )

    return res


if __name__ == "__main__":
    dataset = simpleKISADataLoader('/home/lacie/Datasets/KISA/ver-2/train/')
    print(len(dataset))

    for i in range(len(dataset)):
        data = dataset[i]

        print("Number frames: " + str(len(data['frames'])))
        print("Number human objects: " + str(len(data['human_objects'])))
        print(data['label'])
        print(data['start_time'])
        print(data['duration'])
