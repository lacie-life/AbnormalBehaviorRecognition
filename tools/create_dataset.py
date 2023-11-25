import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import os


def time2second(time_string):
    time_format = "%H:%M:%S"
    time_object = datetime.strptime(time_string, time_format)
    seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
    return seconds


def createLabel(xml_path, file_name, label, output_folder_path, indx):
    # Load the original XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Change the value of the Scenario tag
    for scenario in root.iter('Scenario'):
        scenario.text = label  # Change this to the new value

    # Save the modified content to a new XML file
    tree.write(output_folder_path + f'{label}/' + "/" + file_name + f'segment_{indx}.xml' , encoding='utf-8', xml_declaration=True)
    pass


# Function to split video into 10-second segments
def split_video(video_path, output_folder_path, name, xml_path, label):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    segment_duration = 10  # Split video into 10-second segments

    # Read XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get the event start time and duration
    start_time = root.find(".//StartTime").text
    alarm_duration = root.find(".//AlarmDuration").text

    start_event_frame = time2second(start_time) * fps - 150
    end_event_frame = start_event_frame + time2second(alarm_duration) * fps

    start_frame = 0
    end_frame = int(segment_duration * fps)

    segment_count = 1

    while start_frame < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        type_str = None

        if start_frame > start_event_frame and start_frame < end_event_frame:
            print("Event")
            type_str = label
        else:
            print("Normal")
            type_str = "Normal"

        output_video = cv2.VideoWriter(output_folder_path + f'{type_str}/' + "/" + name + f'segment_{segment_count}.mp4',
                                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

        while start_frame < end_frame and start_frame < total_frames:
            ret, frame = video_capture.read()
            if not ret:
                break
            output_video.write(frame)
            start_frame += 1

        output_video.release()
        createLabel(xml_path, name, type_str, output_folder_path, segment_count)

        segment_count += 1
        end_frame = min(end_frame + int(segment_duration * fps), total_frames)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder_path = '/home/lacie/Datasets/KISA/train/Abandonment/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/Abandonment/'
    label = 'Abandonment'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in range(len(videos)):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)
