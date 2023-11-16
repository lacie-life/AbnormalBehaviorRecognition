import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import os
import os
from pose_estimation import PoseDetector

output_folder_path = "/home/lacie/Datasets/KISA/project/Loitering/"
folder_path = "/home/lacie/Datasets/KISA/train/Loitering"
label = "Loitering"

def time2second(time_string):
    time_format = "%H:%M:%S"
    time_object = datetime.strptime(time_string, time_format)
    seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
    return seconds

def createLabel(label, file_name, areas, output_folder_path, poses = [], boxes = []):
    # Create the root element
    root = ET.Element("KisaLibraryIndex")

    # Create the 'Library' element
    library = ET.SubElement(root, "Library")

    # Add child elements to 'Library'
    ET.SubElement(library, "Scenario").text = label
    ET.SubElement(library, "Dataset").text = "KISA2016"
    ET.SubElement(library, "Libversion").text = "1.0"

    # Create the 'Clip' element
    clip = ET.SubElement(library, "Clip")

    # Create the 'Header' element and add child elements to it
    header = ET.SubElement(clip, "Header")
    ET.SubElement(header, "Filename").text = file_name
    if areas != []:
        areaD = ET.SubElement(header, "Area")
        ET.SubElement(areaD, "Point").text = str(areas[0]) + "," + str(areas[1])
        ET.SubElement(areaD, "Point").text = str(areas[0] + areas[2]) + "," + str(areas[1] + areas[3]) 

    if boxes != []:
        for box in boxes:
            person = ET.SubElement(header, "PersonBox")
            ET.SubElement(person, "BoundingBox").text = str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3])
    
    if poses != []:
        for pose in poses:
            person = ET.SubElement(header, "PersonPose")
            for id, cx, cy in pose:
                ET.SubElement(person, "Point", id=str(id)).text = str(cx) + "," + str(cy)

    file_name = file_name.replace('.jpg', '.xml')

    # Create an ElementTree object and write the XML to a file
    tree = ET.ElementTree(root)
    tree.write(output_folder_path + file_name, encoding="utf-8", xml_declaration=True)

def downsampling(xml_path, video_path):
    
    output_folder_path_frames = output_folder_path + video_path.split("/")[-1].split(".")[0] + "/frames/"
    output_folder_path_labels = output_folder_path + video_path.split("/")[-1].split(".")[0] + "/labels/"
    os.makedirs(output_folder_path_frames, exist_ok=True)
    os.makedirs(output_folder_path_labels, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    start_time = root.find(".//StartTime").text
    alarm_duration = root.find(".//AlarmDuration").text
    detect_area = root.find(".//DetectArea")

    tagArea = ".//" + label
    area = root.find(tagArea)
    abnormal_area = []

    if area is None:
        print("No area")
    else:
        abnormal_area = [tuple(map(int, point.text.split(','))) for point in area.findall("Point")]
        abnormal_area = cv2.boundingRect(np.array(abnormal_area))

    points = [tuple(map(int, point.text.split(','))) for point in detect_area.findall("Point")]

    print("StartTime:", start_time)
    print("AlarmDuration:", alarm_duration)
    print("Detection area points:", points)
    print("Abnormal area:", abnormal_area)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = 30

    # Initialize variables
    frame_count = 0
    success = True

    background = None

    bgSubtractor = cv2.createBackgroundSubtractorMOG2()

    detector = PoseDetector()

    # Loop through the video and extract a frame every second
    while success:
        # Read a frame from the video
        success, frame = cap.read()

        if frame is None:
            break

        detectMask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(detectMask, [np.array(points)], 255)
        _, detectMask = cv2.threshold(detectMask, 1, 255, cv2.THRESH_BINARY)
        frame = cv2.bitwise_and(frame, frame, mask=detectMask)

        if frame_count == time2second(start_time) * fps - 600:
            background = frame

        # Check if we're at a multiple of the frame rate (i.e. every second)
        if frame_count % fps == 0:
            if frame_count >= time2second(start_time)*fps - 100 and frame_count <= (time2second(start_time) + time2second(alarm_duration))*fps:
                
                img, poses, boxes = detector.findPose(frame)
                # Save the frame as an image file
                cv2.imwrite(output_folder_path_frames + "frame%d.jpg" % (int)(frame_count/30), frame)

                createLabel(label, "frame%d.jpg" % (int)(frame_count/30), abnormal_area, output_folder_path_labels, poses, boxes)

                print("Saved frame%d.jpg" % (int)(frame_count/30))
                print("Path:", output_folder_path_frames + "frame%d.jpg" % (int)(frame_count/30))
                print("Label:", output_folder_path_labels + "frame%d.xml" % (int)(frame_count/30))
                
            else:
                # Save the frame as an image file
                cv2.imwrite(output_folder_path_frames + "frame%d.jpg" % (int)(frame_count/30), frame)
                createLabel("Normal", "frame%d.jpg" % (int)(frame_count/30), abnormal_area, output_folder_path_labels)

                print("Saved frame%d.jpg" % (int)(frame_count/30))
                print("Path:", output_folder_path_frames + "frame%d.jpg" % (int)(frame_count/30))
                print("Label:", output_folder_path_labels + "frame%d.xml" % (int)(frame_count/30))


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
    downsampling(xml_file_paths[i], mp4_file_paths[i])





