import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# data_path = '/home/lacie/Videos/data_pose/'
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor(),  # Convert images to PyTorch tensors
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
# ])
#
# # Load the dataset
# dataset = datasets.ImageFolder(root=data_path, transform=transform)
#
# # Determine the lengths of the splits
# train_len = int(0.7 * len(dataset))
# val_len = int(0.15 * len(dataset))
# test_len = len(dataset) - train_len - val_len
#
# # Split the dataset
# train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
#
# # Create DataLoaders for each split
# train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
# #
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Iterate over the DataLoader for training data
# # Load the pre-trained VGG16 model
# model = models.resnext50_32x4d(pretrained=True)
#
# # Freeze the layers
# for param in model.parameters():
#     param.requires_grad = False
#
# # Replace the last layer to match the number of classes in your dataset
# num_classes = len(dataset.classes)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, num_classes)
# )
#
# # Move the model to GPU if available
# model = model.to(device)
#
# # model = SimpleImageClassifier()
# # # Define a loss function and an optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
#
# # Train the model
# for epoch in range(800):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         # print(outputs.shape)
#         # print(labels.shape)
#         # print(outputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         # lr_scheduler.step()
#
#         running_loss += loss.item()
#
#     # print statistics
#     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
#
# print('Finished Training')
#
# # Validate the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in val_loader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f'Accuracy on validation set: {100 * correct / total}%')
#
# # Test the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f'Accuracy on test set: {100 * correct / total}%')
#
# # Save the trained model
# torch.save(model.state_dict(), 'pose_resnext_newdata3_800_epochs.pth')

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnext50_32x4d()
num_classes = 3
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load('/home/lacie/Github/AbnormalBehaviorRecognition/jellyfishABD/pose_resnext_newdata2_200_epochs.pth'))


model.to(device)
#
# # Set the model to evaluation mode
model.eval()

import cv2
import os

img = cv2.imread('/home/lacie/Videos/data_pose/fight/31.jpg')

img_fol = '/home/lacie/Videos/data_pose/walking'

img_path = [os.path.join(img_fol, file) for file in os.listdir(img_fol) if file.endswith(".jpg")]

label = 1
count = 0
for path in img_path:
    # Perform inference on new data
    with torch.no_grad():
        # Assume input_data is your new data for which you want to predict
        img = cv2.imread(path)
        input_data = torch.from_numpy(img).float()
        input_data = input_data.unsqueeze(0)
        input_data = input_data.permute(0, 3, 1, 2)
        input_data = input_data.to(device)
        output = model(input_data)
        predicted = torch.argmax(output, 1)
        print(output.data)
        print(predicted)
        if predicted == label:
            count += 1

# Print the predicted class
print("Accuracy: ", count / len(img_path))




