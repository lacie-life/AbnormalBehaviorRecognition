import torch
import torch.nn as nn
import torch.nn.functional as F
from jellyfishABD import resnext3d


class simpleABD(nn.Module):
    def __init__(self, num_frames, frame_channels, num_classes, num_joints):
        super(simpleABD, self).__init__()

        self.model = resnext3d.resnet50(
            num_classes=256,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            input_channels=frame_channels,
            output_layers=['append'],)


        self.conv2d_bbox = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.linear_layer_bb = nn.Linear(384, 256)

        self.fc_poses_1 = nn.Linear(num_joints, 64)
        self.fc_poses_2 = nn.Linear(64, 128)
        self.fc_poses_3 = nn.Linear(128, 256)
        self.linear_layer_pose = nn.Linear(7680, 256)

        self.fc_fusion_1 = nn.Linear(256 + 256 + 256, 512)
        self.fc_fusion_2 = nn.Linear(512, 256)

        self.fc_output_classes = nn.Linear(256, num_classes)
        self.fc_output_timestamps = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, frames, bounding_boxes, poses):

        # print(bounding_boxes.shape)
        # print(frames.shape)
        # print(poses.shape)

        frames = frames.permute(0, 4, 1, 2, 3)

        x = self.model(frames)

        bounding_boxes = bounding_boxes.view(1, 4, 10, 1)

        bbox_features = self.relu(self.conv2d_bbox(bounding_boxes))
        bbox_features = self.maxpool2d(bbox_features)
        bbox_features = bbox_features.view(bbox_features.size(0), -1)
        bbox_features = self.linear_layer_bb(bbox_features)

        poses_features = self.relu(self.fc_poses_1(poses))
        poses_features = self.relu(self.fc_poses_2(poses_features))
        poses_features = self.relu(self.fc_poses_3(poses_features))
        poses_features = poses_features.view(poses_features.size(0), -1)
        poses_features = self.linear_layer_pose(poses_features)

        # print(x.shape)
        # print(bbox_features.shape)
        # print(poses_features.shape)

        combined_features = torch.cat((x, bbox_features, poses_features), dim=1)
        fused_features = self.dropout(self.relu(self.fc_fusion_1(combined_features)))
        fused_features = self.dropout(self.relu(self.fc_fusion_2(fused_features)))

        event_predictions = self.fc_output_classes(fused_features)
        # timestamps = self.fc_output_timestamps(fused_features)

        max_prob_index = torch.argmax(event_predictions, dim=1)
        selected_event_prediction = event_predictions[0][max_prob_index]
        # selected_timestamps = timestamps[0][max_prob_index]

        # return selected_event_prediction, selected_timestamps
        return event_predictions


# TODO: Testing now
def KISAEvaluationMetric(event_predictions, timestamps, true_event_type, true_start_time, true_duration):
    event_loss = nn.CrossEntropyLoss()(event_predictions, true_event_type)
    timestamp_loss = nn.MSELoss()(timestamps, torch.cat((true_start_time, true_duration), dim=1))

    return event_loss + timestamp_loss
