from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

classifier = SVC(C=3, kernel='rbf', probability=True)

class PoseDataset(Dataset):
    def __init__(self, pose_folder_path):
        self.pose_folder_path = pose_folder_path
        self.file_paths = os.listdir(self.pose_folder_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        self.txt_loader = open(self.file_paths[idx], 'r').readline()
        y = self.txt_loader[0]
        x = []
        for i in range(len(self.txt_loader)):
            if i == 0:
                continue
            x.append(self.txt_loader[i])

        x = torch.tensor(x)

        return x, y

if __name__ == "__main__":
    pose_folder_path = "/home/lacie/Github/AbnormalBehaviorRecognition/final/fall-1.txt"
    pose_dataset = PoseDataset(pose_folder_path)
    pose_dataloader = DataLoader(pose_dataset, batch_size=1, shuffle=True)

    for x, y in pose_dataloader:
        # classifier.fit(x, y)
        # print(classifier.predict(x))
        # print(classifier.predict_proba(x))
        # print(classifier.score(x, y))

        print(x)
        print(y)














