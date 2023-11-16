
import cv2
import xml.etree.ElementTree as ET

# Load image
img = cv2.imread('/home/lacie/Datasets/KISA/project/Abandonment/C002200_002/frames/frame213.jpg')

# Parse XML file
tree = ET.parse('/home/lacie/Datasets/KISA/project/Abandonment/C002200_002/labels/frame213.xml')
root = tree.getroot()

detect_area = root.find(".//Person")

points = [tuple(map(int, point.text.split(','))) for point in detect_area.findall("Point")]

print(points)

for point in points:
    cv2.circle(img, point, 5, (255, 255, 0), cv2.FILLED)

# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
