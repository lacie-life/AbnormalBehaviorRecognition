from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import numpy as np
import pickle
import joblib

keypoints = []
labels = []

input_path = '/home/lacie/Github/AbnormalBehaviorRecognition/final/data_pose/'

data_ls = sorted(os.listdir(input_path))
for file_idx in range (len(data_ls)):
    temp_features = []
    cur_file = open(input_path + data_ls[file_idx], 'r').readlines()
    count = 0

    for line_idx in range (1,len(cur_file)):
        # count=0 means the start of a frame in a video
        if count == 0:
            if cur_file[0] == 'fall\n': labels.append(1)
            elif cur_file[0] == 'fight\n': labels.append(2)
            elif cur_file[0] == 'walk\n': labels.append(0)
        count += 1

        # Check for space position in each line
        for character_idx in range(len(cur_file[line_idx])):
            if cur_file[line_idx][character_idx] == ' ':
                space = character_idx
                break
                
        temp_features.append(float(cur_file[line_idx][0:space]))
        temp_features.append(float(cur_file[line_idx][space+1:-2]))

        # The end of a frame in a video
        if count == 17: 
            keypoints.append(temp_features)
            count=0
            temp_features = []

keypoints = np.array(keypoints)
labels = np.array(labels)

# classifier = svm.SVC(kernel='linear') # Linear Kernel
# classifier.fit(keypoints, labels)
# # svm_weight = pickle.dumps(classifier, 'svm_weight.pkl')
# joblib.dump(classifier, 'svm_weight.pkl')

classifier = joblib.load('svm_weight.pkl')

pred_class = classifier.predict(keypoints[0].reshape(1, -1))
print(pred_class[0])


# print(keypoints[0])
# print(len(keypoints))
# print(len(labels))

