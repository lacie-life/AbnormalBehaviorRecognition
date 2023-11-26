import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm


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

    for old_name in root.iter('Filename'):
        new_name = file_name + '_' + str(indx) + '.mp4'
        old_name.text = str(new_name)  # Change this to the new value

    # Save the modified content to a new XML file
    # with open(xml_name, 'wb+') as xml_file:
    tree.write(output_folder_path + '/' + file_name + '_' + str(indx) + '.xml', encoding='utf-8', xml_declaration=True)


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

    start_event_frame = time2second(start_time) * fps - 200
    end_event_frame = start_event_frame + time2second(alarm_duration) * fps + 100

    start_frame = 0
    end_frame = int(segment_duration * fps)

    segment_count = 1

    # Create the output folder
    os.makedirs(output_folder_path + f'{label}/', exist_ok=True)
    os.makedirs(output_folder_path + 'Normal/', exist_ok=True)
    os.makedirs(output_folder_path + label + '/', exist_ok=True)

    while start_frame < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        type_str = None

        if start_frame > start_event_frame and start_frame < end_event_frame:
            # print("Event")
            type_str = label
        else:
            # print("Normal")
            type_str = "Normal"

        output_video = cv2.VideoWriter(
            output_folder_path + f'{type_str}/' + name + f'_{segment_count}.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

        while start_frame < end_frame and start_frame < total_frames:
            ret, frame = video_capture.read()
            if not ret:
                break
            output_video.write(frame)
            start_frame += 1

        output_video.release()
        createLabel(xml_path, name, type_str, os.path.join(output_folder_path, type_str), segment_count)

        segment_count += 1
        end_frame = min(end_frame + int(segment_duration * fps), total_frames)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder_path_ = '/home/lacie/Datasets/KISA/train/Abandonment/'
    output_folder_path_ = '/home/lacie/Datasets/KISA/ver-3/Abandonment/'
    label = 'Abandonment'

    videos = [file for file in os.listdir(input_folder_path_) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path_) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path_, videos[i])
        xml_path = os.path.join(input_folder_path_, xml_files[i])
        split_video(video_path, output_folder_path_, videos[i][:-4], xml_path, label)

    input_folder_path = '/home/lacie/Datasets/KISA/train/Falldown/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/Falldown/'
    label = 'Falldown'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)

    input_folder_path = '/home/lacie/Datasets/KISA/train/FireDetection/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/FireDetection/'
    label = 'FireDetection'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)

    input_folder_path = '/home/lacie/Datasets/KISA/train/Violence/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/Violence/'
    label = 'Violence'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)

    input_folder_path = '/home/lacie/Datasets/KISA/train/Intrusion/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/Intrusion/'
    label = 'Intrusion'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)

    input_folder_path = '/home/lacie/Datasets/KISA/train/Loitering/'
    output_folder_path = '/home/lacie/Datasets/KISA/ver-3/Loitering/'
    label = 'Loitering'

    videos = [file for file in os.listdir(input_folder_path) if file.endswith('.mp4')]
    xml_files = [file for file in os.listdir(input_folder_path) if file.endswith('.xml')]

    videos.sort()
    xml_files.sort()

    for i in tqdm(range(len(videos))):
        video_path = os.path.join(input_folder_path, videos[i])
        xml_path = os.path.join(input_folder_path, xml_files[i])
        split_video(video_path, output_folder_path, videos[i][:-4], xml_path, label)
