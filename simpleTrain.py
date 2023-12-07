import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from jellyfishABD.simpleABD import simpleABD, KISAEvaluationMetric, MaxProbabilityLoss
from dataset.simpleKISADataLoader import simpleKISADataLoader, collate_fn
from tools.utils import plot_confusion_matrix
import torchsummary as summary
from tqdm import tqdm
import csv
import os

event_list = {'Abandonment': 0,
              'Falldown': 1,
              'FireDetection': 2,
              'Violence': 3}

event_classes = [0, 1, 2, 3]

# Define hyperparameters
num_epochs = 30
learning_rate = 0.001
batch_size = 1

# Define model parameters
frame_channels = 3
num_classes = 4
num_joints = 33
fps = 30
sample = 10
num_frames = fps * 10 // sample

# Initialize the model
model = simpleABD(num_frames, frame_channels, num_classes, num_joints)

model = model.cuda()

# Print model summary
# summary.summary(model, [[30, 3, 224, 224], [30, 4, 1], [30, 3, 33]])

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = MaxProbabilityLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define dataset path
train_path = '/home/lacie/Datasets/KISA/ver-3/4-class/train'
val_path = '/home/lacie/Datasets/KISA/ver-3/4-class/val'

results_path = './results_csn_4_lcass_1/'

if not os.path.exists(results_path):
    os.makedirs(results_path)

# Initialize dataset path
train_dataset = simpleKISADataLoader(train_path, sample=sample, transform=None)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_dataset = simpleKISADataLoader(val_path, sample=sample, transform=None)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

best_metrics = 0.0

print("Data train: " + str(len(train_data_loader)))

log_file = open(results_path + 'log_csn.txt', 'w')
log_file.writelines("Data train: " + str(len(train_data_loader)) + "\n")
log_file.close()

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    # Initialize variables for metrics calculation
    train_total_metrics = 0.0
    train_total_batches = 0

    val_total_metrics = 0.0
    val_total_batches = 0

    train_gt = []
    train_pred = []

    val_gt = []
    val_pred = []

    model.train()

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

        # print(inputs.shape)
        # print(bboxes.shape)
        # print(poses.shape)

        # Forward pass
        # event_predictions, timestamps = model(inputs, bboxes, poses)
        event_predictions = model(inputs, bboxes, poses)

        train_gt.append(event_classes[torch.argmax(true_event_type, dim=1)])
        train_pred.append(event_classes[torch.argmax(event_predictions, dim=1)])

        predict_max = torch.max(event_predictions)

        event_predictions = event_predictions / predict_max

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

            with open(results_path + 'log_csn.txt', 'a') as log_file:
                log_file.writelines(f"Epoch [{epoch + 1}/{num_epochs}], "
                                    f"Batch [{i + 1}/{len(train_data_loader)}], "
                                    f"Loss: {running_loss / 10:.4f}\n")
                log_file.close()
            running_loss = 0.0

        metrics = loss.item()

        train_total_metrics += metrics
        train_total_batches += 1

    # Calculate average metrics for the epoch
    train_average_metrics = train_total_metrics / train_total_batches
    print("Training Metrics:", train_average_metrics)

    torch.save(model.state_dict(), f'model_csn_{epoch}.pt')

    model.eval()

    # Validation loop
    with torch.no_grad():
        for i, (video_frames, bboxes, poses, event_label, start_time, duration) in tqdm(enumerate(val_data_loader)):
            if video_frames.shape[1] < fps * 10 / sample:
                continue
            val_inputs, val_labels = video_frames.cuda(), event_label.cuda()
            val_bboxes, val_poses = bboxes.cuda(), poses.cuda()
            val_true_start_time, val_true_event_type = start_time.cuda(), event_label.cuda()
            val_true_duration = duration.cuda()

            # Forward pass
            val_event_predictions = model(val_inputs, val_bboxes, val_poses)

            # Calculate evaluation metrics
            val_pred_event_type = torch.argmax(val_event_predictions, dim=1)

            val_gt.append(event_classes[torch.argmax(val_true_event_type, dim=1)])
            val_pred.append(event_classes[torch.argmax(val_event_predictions, dim=1)])

    plot_confusion_matrix(val_gt, val_pred, event_classes, results_path + f'confusion_matrix_{epoch}/', f'confusion_matrix_val_{epoch}.png')
    plot_confusion_matrix(train_gt, train_pred, event_classes, results_path + f'confusion_matrix_{epoch}/', f'confusion_matrix_train_{epoch}.png')

    # Save the model if it has the best metrics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Metrics: {train_average_metrics}")
    if train_average_metrics > best_metrics:
        torch.save(model.state_dict(), results_path + f'best_model_{epoch}.pt')
        best_metrics = train_average_metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")
    else:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Best Metrics: {best_metrics}")

print('Finished Training')


