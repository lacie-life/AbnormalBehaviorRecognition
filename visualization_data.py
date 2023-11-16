
import cv2
import xml.etree.ElementTree as ET

# Load image
img = cv2.imread('/home/lacie/Datasets/KISA/project/Loitering/C001101_003/frames/frame14.jpg')

# Parse XML file
tree = ET.parse('/home/lacie/Datasets/KISA/project/Loitering/C001101_003/labels/frame14.xml')
root = tree.getroot()

detect_area = root.find(".//Area")

points = [tuple(map(int, point.text.split(','))) for point in detect_area.findall("Point")]

# Draw bounding box on image
cv2.rectangle(img, points[0], points[1], (255, 255, 0), 2)

# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
