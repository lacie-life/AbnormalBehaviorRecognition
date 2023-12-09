import xml.etree.ElementTree as ET

class XMLInfo:
    def __init__(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        self.numberArea = int(self.root.find(".//DetectionAreas").text)

        self.areas = []
        self.abnormal_area = []
        self.abnormal_type = None
        self.video_path = None

        if self.root.find(".//Filename") is not None:
            path = xml_path.replace(".xml", ".mp4")
            self.video_path = path

        if self.numberArea == 1:
            area = self.root.find(".//DetectArea")
            self.areas = [tuple(map(int, point.text.split(','))) for point in area.findall("Point")]
        if self.numberArea == 2:
            area = self.root.find(".//DetectArea")
            self.areas = [tuple(map(int, point.text.split(','))) for point in area.findall("Point")]
            if self.root.find(".//Intrusion") is not None:
                abnormal_area = self.root.find(".//Intrusion")
                self.abnormal_area = [tuple(map(int, point.text.split(','))) for point in abnormal_area.findall("Point")]
                self.abnormal_type = 'Intrusion'
            if self.root.find(".//Loitering") is not None:
                abnormal_area = self.root.find(".//Loitering")
                self.abnormal_area = [tuple(map(int, point.text.split(','))) for point in abnormal_area.findall("Point")]
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
        return data
