import cv2
import xml.etree.ElementTree as ET
from func import XMLInfo
import os

data_folder_path = "/home/lacie/Datasets/KISA/train/FireDetection"

def get_video_infor(data_folder_path):
    xml_paths = [os.path.join(data_folder_path, file) for file in os.listdir(data_folder_path) if file.endswith(".xml")]
    video_infor = []
    for xml_path in xml_paths:
        xml_info = XMLInfo(xml_path)
        video_infor.append(xml_info.get_video_infor())
    return video_infor

if __name__ == "__main__":
    video_infor = get_video_infor(data_folder_path)
    for infor in video_infor:
        print(infor)

