import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import os
import os

output_folder_path = "/home/lacie/Github/AbnormalBehaviorRecognition/data/kisa/abandonment/"
folder_path = "/home/lacie/Datasets/KISA/train/Abandonment"

def time2second(time_string):
    time_format = "%H:%M:%S"
    time_object = datetime.strptime(time_string, time_format)
    seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
    return seconds

def downsampling(xml_path, video_path, output_folder_path):
    
    output_folder_path = output_folder_path + video_path.split("/")[-1].split(".")[0] + "/frames/"
    os.makedirs(output_folder_path, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    start_time = root.find(".//StartTime").text
    alarm_duration = root.find(".//AlarmDuration").text

    print("StartTime:", start_time)
    print("AlarmDuration:", alarm_duration)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = 30

    # Initialize variables
    frame_count = 0
    success = True

    background = None

    bgSubtractor = cv2.createBackgroundSubtractorMOG2()

    # Loop through the video and extract a frame every second
    while success:
        # Read a frame from the video
        success, frame = cap.read()

        if frame is None:
            break

        if frame_count == time2second(start_time) * fps - 600:
            background = frame

        # Check if we're at a multiple of the frame rate (i.e. every second)
        if frame_count % fps == 0:
            if frame_count >= time2second(start_time)*fps - 600 and frame_count <= (time2second(start_time) + time2second(alarm_duration))*fps:
                # fgMask = np.zeros_like(frame)
                # diff = cv2.absdiff(background, frame)
                # diff = cv2.convertScaleAbs(diff)

                # # fbMask = bgSubtractor.apply(frame)

                # if len(fgMask.shape) > 2 or fgMask.dtype != np.uint8:
                #     fgMask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                #     _, fgMask = cv2.threshold(fgMask, 30, 255, cv2.THRESH_BINARY)

                # masked_image = cv2.bitwise_and(frame, frame, mask=fgMask)

                # Save the frame as an image file
                cv2.imwrite(output_folder_path + "frame%d.jpg" % (int)(frame_count/30), frame)
                print("Saved frame%d.jpg" % (int)(frame_count/30))
                print("Path:", output_folder_path + "frame%d.jpg" % (int)(frame_count/30))
            else:
                # Save the frame as an image file
                cv2.imwrite(output_folder_path + "frame%d.jpg" % (int)(frame_count/30), frame)
                print("Saved frame%d.jpg" % (int)(frame_count/30))
                print("Path:", output_folder_path + "frame%d.jpg" % (int)(frame_count/30))

        # Increment the frame count
        frame_count += 1

    # Release the video file
    cap.release()


# Get a list of all the files in the folder
files = os.listdir(folder_path)
# Filter the list to only include .mp4 and .xml files
mp4_files = [f for f in files if f.endswith(".mp4")]
xml_files = [f for f in files if f.endswith(".xml")]
# Sort the list alphabetically
mp4_files.sort()
xml_files.sort()
# Create a list of file paths
mp4_file_paths = [os.path.join(folder_path, f) for f in mp4_files]
xml_file_paths = [os.path.join(folder_path, f) for f in xml_files]
print(mp4_file_paths)
print(xml_file_paths)

for i in range(len(mp4_file_paths)):
    downsampling(xml_file_paths[i], mp4_file_paths[i], output_folder_path)





