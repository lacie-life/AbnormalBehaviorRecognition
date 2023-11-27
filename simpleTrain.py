import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from jellyfishABD.simpleABD import simpleABD, KISAEvaluationMetric
from dataset.simpleKISADataLoader import simpleKISADataLoader, collate_fn
import torchsummary as summary
from tqdm import tqdm
import csv

event_list = {'Abandonment': 0,
              'Falldown': 1,
              'FireDetection': 2,
              'Intrusion': 3,
              'Loitering': 4,
              'Violence': 5,
              'Normal': 6}


# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 1

# Define model parameters
frame_channels = 3
num_classes = 7
num_joints = 33
fps = 30
sample = 10
num_frames = fps * 10 // sample

# Initialize the model
model = simpleABD(num_frames, frame_channels, num_classes, num_joints)

model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define dataset path
train_path = '/home/lacie/Datasets/KISA/ver-3/train'
val_path = '/home/lacie/Datasets/KISA/ver-3/val'

# Initialize dataset path
train_dataset = simpleKISADataLoader(train_path, sample=sample, transform=None)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_dataset = simpleKISADataLoader(val_path, sample=sample, transform=None)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

best_metrics = 0.0

print("Data train: " + str(len(train_data_loader)))

# Initialize log file
log_file = open('log.csv', 'w')

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    # Initialize variables for metrics calculation
    train_total_metrics = 0.0
    train_total_batches = 0

    val_total_metrics = 0.0
    val_total_batches = 0

    for i, (video_frames, bboxes, poses, event_label, start_time, duration) in tqdm(enumerate(train_data_loader)):
        if video_frames.shape[1] < fps * 10 / sample:
            continue
        inputs = video_frames.float().cuda()
        labels = event_label.float().cuda()
        bboxes = bboxes.float().cuda()
        poses = poses.float().cuda()
        true_start_time, true_event_type = start_time.cuda(), event_label.cuda()
        true_duration = duration.cuda()

        optimizer.zero_grad()

        # Forward pass
        # event_predictions, timestamps = model(inputs, bboxes, poses)
        event_predictions = model(inputs, bboxes, poses)

        # print(event_predictions)
        # print(true_event_type)

        # Compute loss
        loss = criterion(event_predictions, true_event_type)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Batch [{i + 1}/{len(train_data_loader)}], "
                  f"Loss: {running_loss / 10:.4f}")

            log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], "
                           f"Batch [{i + 1}/{len(train_data_loader)}], "
                           f"Loss: {running_loss / 10:.4f}\n")
            running_loss = 0.0


        # Calculate evaluation metrics
        # pred_event_type = torch.argmax(event_predictions, dim=1)
        # pred_start_time = timestamps[:, 0]
        # pred_duration = timestamps[:, 1]

        metrics = loss.item()

        train_total_metrics += metrics
        train_total_batches += 1

    # Calculate average metrics for the epoch
    train_average_metrics = train_total_metrics / train_total_batches
    print("Training Metrics:", train_average_metrics)

    # Validation loop
    # with torch.no_grad():
    #     for i, (video_frames, bboxes, poses, event_label, start_time, duration) in tqdm(enumerate(val_data_loader)):
    #         val_inputs, val_labels = video_frames.cuda(), event_label.cuda()
    #         val_bboxes, val_poses = bboxes.cuda(), poses.cuda()
    #         val_true_start_time, val_true_event_type = start_time.cuda(), event_label.cuda()
    #         val_true_duration = duration.cuda()
    #
    #         # Forward pass
    #         val_event_predictions, val_timestamps = model(val_inputs, val_bboxes, val_poses)
    #
    #         # Calculate evaluation metrics
    #         val_pred_event_type = torch.argmax(val_event_predictions, dim=1)
    #
    #         val_total_metrics += metrics
    #         val_total_batches += 1

    # # Calculate average metrics for the epoch
    # val_average_metrics = val_total_metrics / val_total_batches
    # print("Validation Metrics:", val_average_metrics)

    # Save the model if it has the best metrics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Metrics: {train_average_metrics}")
    if train_average_metrics > best_metrics:
        torch.save(model.state_dict(), 'best_model.pt')
        best_metrics = train_average_metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")
    else:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")



