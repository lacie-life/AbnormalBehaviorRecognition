from skimage.feature import hog
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cv2
import os

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import pickle


Categories = ['fall', 'fight', 'walking']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = '/home/lacie/Github/data_pose/'
# path which contains all the categories of images
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# dataframe
df = pd.DataFrame(flat_data)
df['Target'] = target
df.shape

# input data
x = df.iloc[:, :-1]
# output data
y = df.iloc[:, -1]


hog_features_arr = []
for img in flat_data_arr:
    img_gray = rgb2gray(img.reshape(150, 150, 3))
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hog_features_arr.append(hog_features)

hog_features = np.array(hog_features_arr)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, target, test_size=0.2, random_state=42)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Evaluate the classifier
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred, target_names=Categories))

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

def predict_image(image_path):
    # Load the saved model
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)

    # Load the image and extract HOG features
    img = imread(image_path)
    img_gray = rgb2gray(img.reshape(150, 150, 3))
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Predict the class of the image
    prediction = knn.predict([hog_features])

    return prediction

