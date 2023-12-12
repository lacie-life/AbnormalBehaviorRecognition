import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

data_path = '/home/lacie/Github/AbnormalBehaviorRecognition/final/data_pose'

# Load the dataset
dataset = datasets.ImageFolder(root=data_path, transform=transforms.ToTensor())

# Determine the lengths of the splits
train_len = int(0.7 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len

# Split the dataset
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

# Create DataLoaders for each split
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Iterate over the DataLoader for training data
# Load the pre-trained VGG16 model
model = models.resnext50_32x4d(pretrained=True)

# Freeze the layers
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer to match the number of classes in your dataset
num_classes = len(dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(50):  # loop over the dataset multiple times
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
torch.save(model.state_dict(), 'pose_resnext.pth')
#
# # Load the saved model
# model = models.vgg16()  # we initialize the model first
# model.load_state_dict(torch.load('model_vgg16.pth'))
# model = model.to(device)  # move the model to the device
#
# # Set the model to evaluation mode
# model.eval()
#
# # Perform inference on new data
# with torch.no_grad():
#     # Assume input_data is your new data for which you want to predict
#     input_data = input_data.to(device)
#     output = model(input_data)
#     _, predicted = torch.max(output.data, 1)
#
# # Print the predicted class
# print(f'Predicted class: {predicted.item()}') model