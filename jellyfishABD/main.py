import cv2
import xml.etree.ElementTree as ET
from ILDetector import XMLInfo, ILDetector
from SimpleABDetector import SimpleABDetector
import os

data_folder_path = "/home/lacie/Datasets/KISA/train/Loitering/test"
output_folder_path = "/home/lacie/Datasets/KISA/results/output"
pre_train_path = "/home/lacie/Github/AbnormalBehaviorRecognition/pre-train"

def get_video_infor(data_folder_path, output_folder_path):
    xml_paths = [os.path.join(data_folder_path, file) for file in os.listdir(data_folder_path) if file.endswith(".xml")]
    video_infor = []
    for xml_path in xml_paths:
        xml_info = XMLInfo(xml_path, output_folder_path)
        video_infor.append(xml_info.get_video_infor())
    return video_infor


if __name__ == "__main__":
    video_infor = get_video_infor(data_folder_path, output_folder_path)
    for infor in video_infor:
        print(infor)
        if infor["abnormal_type"] is not None:
            detector = ILDetector(infor, pretrain_path=pre_train_path, visual=True)
            detector.process_video()
        else:
            detector = SimpleABDetector(infor, pretrain_path=pre_train_path)
            detector.process_video()


