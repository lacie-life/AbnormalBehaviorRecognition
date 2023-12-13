import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

data_path = '/home/lacie/Github/AbnormalBehaviorRecognition/data/data_pose/images'

# Load the dataset
dataset = datasets.ImageFolder(root=data_path, transform=transforms.ToTensor())

# Determine the lengths of the splits
train_len = int(0.7 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len

# Split the dataset
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

# Create DataLoaders for each split
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over the DataLoader for training data
# Load the pre-trained VGG16 model
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

# defining the model architecture
class SimpleImageClassifier(nn.Module):
    def __init__(self):
        super(SimpleImageClassifier, self).__init__()

        self.layer1 = self.ConvModule(in_features=3, out_features=64)  # 112,112
        self.layer2 = self.ConvModule(in_features=64, out_features=128)  # 56,56
        self.layer3 = self.ConvModule(in_features=128, out_features=256)  # 28,28
        self.layer4 = self.ConvModule(in_features=256, out_features=512)  # 14,14
        self.layer5 = self.ConvModule(in_features=512, out_features=512)  # 7,7

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(7 * 7 * 512, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 2),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x

    def ConvModule(self, in_features, out_features):
        return nn.Sequential(nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_features),
                             nn.ReLU(),
                             nn.MaxPool2d(2, 2)
                             )

model = SimpleImageClassifier()
# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# Train the model
for epoch in range(400):  # loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print statistics
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')

# Validate the model
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on validation set: {100 * correct / total}%')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')

# Save the trained model
torch.save(model.state_dict(), 'pose_resnext_400_epochs.pth')

# # Load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnext50_32x4d()
# num_classes = 3
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, num_classes)
# )
# model.load_state_dict(torch.load('/home/lacie/Github/AbnormalBehaviorRecognition/jellyfishABD/pose_resnext_400_epochs.pth'))
#
#
# model.to(device)
#
# # Set the model to evaluation mode
# model.eval()
#
# import cv2
#
# img = cv2.imread('/home/lacie/Github/AbnormalBehaviorRecognition/data/data_pose/images/fight/63.jpg')
#
# # Perform inference on new data
# with torch.no_grad():
#     # Assume input_data is your new data for which you want to predict
#     input_data = torch.from_numpy(img).float()
#     input_data = input_data.unsqueeze(0)
#     input_data = input_data.permute(0, 3, 1, 2)
#     input_data = input_data.to(device)
#     output = model(input_data)
#     _, predicted = torch.max(output.data, 1)
#     print(output.data)
#     print(predicted)
#
# # Print the predicted class
# print(f'Predicted class: {predicted.item()}')




