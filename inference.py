import torch
import cv2
import torchvision.transforms as transforms
from jellyfishABD.simpleABD import simpleABD

# Path to your video file
video_path = './test/video.mp4'

# Define model parameters
num_frames = 60
frame_channels = 3
num_classes = 7
num_joints = 17

# Initialize the model and load trained weights
model = simpleABD(num_frames, frame_channels, num_classes, num_joints)  # Replace with your model architecture
model.load_state_dict(torch.load('./weights/model_weights.pth'))
model.eval()

# Open the video file
video_capture = cv2.VideoCapture(video_path)

input_frames = []

frame_count = 0

# Loop through video frames for get the input data
while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if there are no more frames

    # Perform preprocessing on the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    if frame_count % 10 == 0:
        input_frames.append(frame)
    frame_count += 1

video_capture.release()

# Perform inference
# with torch.no_grad():
#     val_event_predictions, val_timestamps = model(val_inputs, val_bboxes, val_poses)

