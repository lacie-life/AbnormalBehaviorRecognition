import torch
import cv2
import torchvision.transforms as transforms
from jellyfishABD.simpleABD import simpleABD
from tools.pose_estimation import PoseDetector

# Path to your video file
video_path = './test/video.mp4'

# Define model parameters
num_frames = 60
frame_channels = 3
num_classes = 7
num_joints = 33

# Initialize the model and load trained weights
model = simpleABD(num_frames, frame_channels, num_classes, num_joints)  # Replace with your model architecture
model.load_state_dict(torch.load('./weights/model_weights.pth'))
model.eval()

# Open the video file
video_capture = cv2.VideoCapture(video_path)

fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

input_frames = []

frame_count = 0

step = 60

start_frame = 0
end_frame = int(step * fps)

segment_video = []

# Loop through video frames for get the input data
while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if there are no more frames

    # Perform preprocessing on the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    while start_frame < end_frame and start_frame < total_frames:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_LINEAR)
        segment_video.append(frame)
        start_frame += 1

    input_frames.append(segment_video)
    segment_video.clear()
    end_frame = min(end_frame + int(step * fps), total_frames)

video_capture.release()

# Processing

# Initialize the pose detector
pose_detector = PoseDetector()
video_summary = []

for seg in input_frames:
    # Extract human information from the video frame
    bboxes, poses = pose_detector.findPoseMultiFrame(seg)

    # Perform preprocessing on the frame
    seg = [transforms.ToTensor()(frame) for frame in seg]  # Convert to tensor
    seg = torch.stack(seg)  # Stack into 4D tensor
    seg = seg.permute(3, 0, 1, 2)  # Change the order of the dimensions

    # Feed the input to the model
    with torch.no_grad():
        output = model(seg, bboxes, poses)

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Get the predicted class label
    predicted_label = predicted.item()

    # Get the predicted class probability
    predicted_prob = torch.nn.functional.softmax(output.data, dim=1)[0][predicted_label].item()

    # Append the result to the video summary
    video_summary.append((predicted_label, predicted_prob))

# Print the video summary
print(video_summary)

# TODO: Extract start time and duration


