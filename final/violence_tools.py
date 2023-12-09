import cv2
import numpy as np
import os

images = []

# Read all image in folder
image_folder = '/home/lacie/Datasets/KISA/project/Falldown/C004200_005/frames/Falldown'
image_files = os.listdir(image_folder)
image_files = [img for img in image_files if img.endswith(".jpg") or img.endswith(".png")]

image_files.sort()

for image_file in image_files:
    image = cv2.imread(os.path.join(image_folder, image_file))
    images.append(image)

# video_path = '/home/lacie/Videos/vlc-record-2023-12-09-22h49m06s-C003200_001.mp4-.mp4'
# cap = cv2.VideoCapture(video_path)
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow("frame", frame)
#         cv2.waitKey(1)
#
#         if frame_count % 10 == 0:
#             images.append(frame)
#     else:
#         break


for i in range(len(images)):

    # Subtract the two images
    difference = cv2.subtract(images[i], images[i+1])

    img_1 = cv2.resize(images[i], (0, 0), None, 0.5, 0.5)
    img_2 = cv2.resize(images[i+1], (0, 0), None, 0.5, 0.5)
    difference = cv2.resize(difference, (0, 0), None, 0.5, 0.5)

    img_3 = np.vstack([img_1, img_2, difference])
    cv2.imwrite(f"./output/test{i}.jpg", img_3)
    # Show 1 window for all 3 images
    cv2.imshow("Images", img_3)

    cv2.waitKey(0)

