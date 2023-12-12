import torch
import torch.nn as nn
import torch.nn.functional as F
from jellyfishABD import resnext3d, csn, resnet3d


class simpleABD(nn.Module):
    def __init__(self, num_frames, frame_channels, num_classes, num_joints):
        super(simpleABD, self).__init__()

        self.num_frames = num_frames

        # self.model = resnext3d.resnet50(
        #     num_classes=256,
        #     shortcut_type='B',
        #     cardinality=32,
        #     sample_size=112,
        #     sample_duration=16,
        #     input_channels=frame_channels,
        #     output_layers=['append'],
        #     number_frames=num_frames)
        
        self.model = csn.csn101(num_classes=256, mode='ip')

        # self.model = resnet3d.resnet101(num_classes=256)


        self.conv2d_bbox = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.linear_layer_bb = nn.Linear(1024, 256)

        self.fc_poses_1 = nn.Linear(num_joints, 64)
        self.fc_poses_2 = nn.Linear(64, 128)
        self.fc_poses_3 = nn.Linear(128, 256)
        self.linear_layer_pose = nn.Linear(self.num_frames * frame_channels * 256, 256)

        self.fc_fusion_1 = nn.Linear(256 + 256 + 256, 512)
        self.fc_fusion_2 = nn.Linear(512, 256)

        self.fc_output_classes = nn.Linear(256, num_classes)
        self.fc_output_timestamps = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, frames, bounding_boxes, poses):

        # print(bounding_boxes.shape)
        # print(frames.shape)
        # print(poses.shape)

        frames = frames.permute(0, 2, 1, 4, 3)
        bounding_boxes = bounding_boxes.permute(0, 2, 1, 3)

        x = self.model(frames)

        bbox_features = self.relu(self.conv2d_bbox(bounding_boxes))
        bbox_features = self.maxpool2d(bbox_features)
        bbox_features = bbox_features.view(bbox_features.size(0), -1)
        bbox_features = self.linear_layer_bb(bbox_features)

        poses_features = self.relu(self.fc_poses_1(poses))
        poses_features = self.relu(self.fc_poses_2(poses_features))
        poses_features = self.relu(self.fc_poses_3(poses_features))
        poses_features = poses_features.view(poses_features.size(0), -1)
        poses_features = self.linear_layer_pose(poses_features)

        combined_features = torch.cat((x, bbox_features, poses_features), dim=1)
        fused_features = self.dropout(self.relu(self.fc_fusion_1(combined_features)))
        fused_features = self.dropout(self.relu(self.fc_fusion_2(fused_features)))

        event_predictions = self.relu(self.fc_output_classes(fused_features))

        # event_predictions = self.sigmoid(event_predictions)

        # max_prob_index = torch.argmax(event_predictions, dim=1)
        # selected_event_prediction = event_predictions[0][max_prob_index]

        # return selected_event_prediction
        return event_predictions


# TODO: Testing now
def KISAEvaluationMetric(event_predictions, timestamps, true_event_type, true_start_time, true_duration):
    event_loss = nn.CrossEntropyLoss()(event_predictions, true_event_type)
    timestamp_loss = nn.MSELoss()(timestamps, torch.cat((true_start_time, true_duration), dim=1))

    return event_loss + timestamp_loss


class MaxProbabilityLoss(nn.Module):
    def __init__(self):
        super(MaxProbabilityLoss, self).__init__()

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)  # Calculate cross-entropy loss

        # Find the index of the maximum probability class
        max_prob_index = torch.argmax(outputs, dim=1)

        # Create a mask to identify correct predictions (1 for correct, 0 for incorrect)
        correct_predictions = (max_prob_index == targets).float()

        # Modify the loss to penalize only incorrect predictions
        modified_loss = (1.0 - correct_predictions) * ce_loss

        return torch.mean(modified_loss)
