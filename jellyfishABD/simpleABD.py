import torch
import torch.nn as nn


class simpleABD(nn.Module):
    def __init__(self, num_frames, frame_channels, num_classes, num_joints):
        super(simpleABD, self).__init__()
        self.conv3d_1 = nn.Conv3d(frame_channels, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2d_bbox = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc_poses_1 = nn.Linear(num_joints * 2, 128)
        self.fc_poses_2 = nn.Linear(128, 256)

        self.fc_fusion_1 = nn.Linear(256 + 256 + 256, 512)
        self.fc_fusion_2 = nn.Linear(512, 256)

        self.fc_output_classes = nn.Linear(256, num_classes)
        self.fc_output_timestamps = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, frames, bounding_boxes, poses):
        x = self.relu(self.conv3d_1(frames))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_2(x))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_3(x))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_4(x))
        x = self.maxpool3d(x)
        x = x.view(x.size(0), -1)

        if bounding_boxes.size() == 0:
            bounding_boxes = torch.zeros((1, 4, 4, 4))

        if poses.size() == 0:
            poses = torch.zeros((1, 34))

        bbox_features = self.relu(self.conv2d_bbox(bounding_boxes))
        bbox_features = self.maxpool2d(bbox_features)
        bbox_features = bbox_features.view(bbox_features.size(0), -1)

        poses_features = self.relu(self.fc_poses_1(poses))
        poses_features = self.relu(self.fc_poses_2(poses_features))

        combined_features = torch.cat((x, bbox_features, poses_features), dim=1)
        fused_features = self.dropout(self.relu(self.fc_fusion_1(combined_features)))
        fused_features = self.dropout(self.relu(self.fc_fusion_2(fused_features)))

        event_predictions = self.fc_output_classes(fused_features)
        timestamps = self.fc_output_timestamps(fused_features)

        max_prob_index = torch.argmax(event_predictions, dim=1)
        selected_event_prediction = event_predictions[0][max_prob_index]
        selected_timestamps = timestamps[0][max_prob_index]

        return selected_event_prediction, selected_timestamps


# TODO: Testing now
def KISALoss(event_predictions, timestamps, true_event_type, true_start_time, true_duration):
    event_loss = nn.CrossEntropyLoss()(event_predictions, true_event_type)
    timestamp_loss = nn.MSELoss()(timestamps, torch.cat((true_start_time, true_duration), dim=1))

    return event_loss + timestamp_loss
