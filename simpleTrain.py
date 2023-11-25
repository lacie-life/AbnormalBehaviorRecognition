import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from jellyfishABD.simpleABD import simpleABD, KISALoss
from dataset.simpleKISADataLoader import simpleKISADataLoader
import torchsummary as summary


# Define hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 1

# Define model parameters
num_frames = 60
frame_channels = 3
num_classes = 7
num_joints = 17

# Initialize the model
model = simpleABD(num_frames, frame_channels, num_classes, num_joints)

model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define dataset path
train_path = '/home/lacie/Datasets/KISA/ver-2/train'
val_path = '/home/lacie/Datasets/KISA/ver-2/val'

# Initialize dataset path
train_dataset = simpleKISADataLoader(train_path, transform=None)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = simpleKISADataLoader(val_path, transform=None)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

best_metrics = 0.0

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    # Initialize variables for metrics calculation
    total_metrics = 0.0
    total_batches = 0

    for i, data in enumerate(train_data_loader):
        inputs, labels = data['frames'], data['label']
        bboxes, poses = data['bounding_boxes'], data['poses']
        true_start_time, true_event_type = data['start_time'], data['event_type']
        true_duration = data['duration']

        print("Batch:", i)
        print(inputs.shape)
        print(labels.shape)
        print(bboxes.shape)
        print(poses.shape)
        print(true_start_time.shape)
        print(true_event_type.shape)
        print(true_duration.shape)

        if i >= 2:
            break

    #     optimizer.zero_grad()
    #
    #     # Forward pass
    #     event_predictions, timestamps = model(inputs, bboxes, poses)
    #
    #     # Compute loss
    #     loss = criterion(event_predictions, labels)
    #
    #     # Backward pass and optimize
    #     loss.backward()
    #     optimizer.step()
    #
    #     # Print statistics
    #     running_loss += loss.item()
    #     if i % 10 == 9:  # Print every 10 mini-batches
    #         print(f"Epoch [{epoch + 1}/{num_epochs}], "
    #               f"Batch [{i + 1}/{len(train_data_loader)}], "
    #               f"Loss: {running_loss / 10:.4f}")
    #         running_loss = 0.0
    #
    #     # Validation loop
    #     # with torch.no_grad():
    #     #     for val_data in val_data_loader:
    #     #         val_inputs, val_labels = val_data['frames'], val_data['label']
    #     #         val_bboxes, val_poses = val_data['bounding_boxes'], val_data['poses']
    #     #         val_true_start_time, val_true_event_type = val_data['start_time'], val_data['event_type']
    #     #
    #     #         # Forward pass
    #     #         val_event_predictions, val_timestamps = model(val_inputs, val_bboxes, val_poses)
    #
    #     # Calculate evaluation metrics
    #     pred_event_type = torch.argmax(event_predictions, dim=1)
    #     pred_start_time = timestamps[:, 0]
    #     pred_duration = timestamps[:, 1]
    #
    #     metrics = KISALoss(pred_event_type, timestamps,
    #                        true_event_type, true_start_time, true_duration)
    #
    #     total_metrics += metrics
    #     total_batches += 1
    #
    # # Calculate average metrics for the epoch
    # average_metrics = total_metrics / total_batches
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Average Metrics: {average_metrics}")
    #
    # if average_metrics > best_metrics:
    #     torch.save(model.state_dict(), 'best_model.pt')
    #     best_metrics = average_metrics
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")
    # else:
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")



