import torch
import torch.nn as nn
import torch.nn.functional as F
from jellyfishABD.resnext3d import get_fine_tuning_parameters
from jellyfishABD import resnext3d


class simpleABD(nn.Module):
    def __init__(self, num_frames, frame_channels, num_classes, num_joints, opt):
        super(simpleABD, self).__init__()
        # self.conv3d_1 = nn.Conv3d(frame_channels, 32, kernel_size=(3, 3, 3), padding=1)
        # self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        # self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        # self.conv3d_4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        # self.conv3d_5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        # self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.linear_layer_img = nn.Linear(921600, 256).cuda()

        self.model, _ = resnext3d.resnet50(
                        num_classes=opt.n_classes,
                        shortcut_type=opt.resnet_shortcut,
                        cardinality=opt.resnext_cardinality,
                        sample_size=opt.sample_size,
                        sample_duration=opt.sample_duration,
                        input_channels=opt.input_channels,
                        output_layers=opt.output_layers)

        self.conv2d_bbox = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, padding=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.linear_layer_bb = nn.Linear(1024, 256).cuda()

        self.fc_poses_1 = nn.Linear(num_joints, 64)
        self.fc_poses_2 = nn.Linear(64, 128)
        self.fc_poses_3 = nn.Linear(128, 256)
        # self.linear_layer_pose = nn.Linear(256, 256).cuda()

        self.fc_fusion_1 = nn.Linear(256 + 256 + 256, 512)
        self.fc_fusion_2 = nn.Linear(512, 256)

        self.fc_output_classes = nn.Linear(256, num_classes)
        self.fc_output_timestamps = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, frames, bounding_boxes, poses):

        frames = frames.permute(0, 4, 1, 2, 3)

        x = self.relu(self.conv3d_1(frames))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_2(x))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_3(x))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_4(x))
        x = self.maxpool3d(x)
        x = self.relu(self.conv3d_5(x))
        x = self.maxpool3d(x)
        x = x.view(x.size(0), -1)

        # x = x.reshape(x.size(0), -1)
        # linear_layer_img = nn.Linear(x.size(1), 256).cuda()
        # x = x.view(1, -1)
        # x = linear_layer_img(x)


        bounding_boxes = bounding_boxes.view(1, 4, 30, 1)
        bbox_features = self.relu(self.conv2d_bbox(bounding_boxes))
        bbox_features = self.maxpool2d(bbox_features)
        # bbox_features = bbox_features.view(bbox_features.size(0), -1)
        # linear_layer_bb = nn.Linear(bbox_features.size(1), 256).cuda()
        # bbox_features = bbox_features.view(1, -1)
        # bbox_features = linear_layer_bb(bbox_features)

        poses_features = self.relu(self.fc_poses_1(poses))
        poses_features = self.relu(self.fc_poses_2(poses_features))
        poses_features = self.relu(self.fc_poses_3(poses_features))
        # poses_features = poses_features.view(poses_features.size(0), -1)
        # linear_layer_pose = nn.Linear(poses_features.size(1), 256).cuda()
        # poses_features = poses_features.view(1, -1)
        # poses_features = linear_layer_pose(poses_features)

        print(x.shape)
        print(bbox_features.shape)
        print(poses_features.shape)

        combined_features = torch.cat((x, bbox_features, poses_features), dim=1)
        fused_features = self.dropout(self.relu(self.fc_fusion_1(combined_features)))
        fused_features = self.dropout(self.relu(self.fc_fusion_2(fused_features)))

        event_predictions = self.fc_output_classes(fused_features)
        timestamps = self.fc_output_timestamps(fused_features)

        max_prob_index = torch.argmax(event_predictions, dim=1)
        selected_event_prediction = event_predictions[0][max_prob_index]
        selected_timestamps = timestamps[0][max_prob_index]

        return selected_event_prediction, selected_timestamps
        # return selected_event_prediction


# TODO: Testing now
def KISAEvaluationMetric(event_predictions, timestamps, true_event_type, true_start_time, true_duration):
    event_loss = nn.CrossEntropyLoss()(event_predictions, true_event_type)
    timestamp_loss = nn.MSELoss()(timestamps, torch.cat((true_start_time, true_duration), dim=1))

    return event_loss + timestamp_loss
