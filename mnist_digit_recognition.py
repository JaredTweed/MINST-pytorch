import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Enhanced Neural Network
class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        # First Convolutional Block with Residual Connection
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        # Second Convolutional Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv2(F.leaky_relu(self.conv1(x)))))
        x = F.leaky_relu(self.bn2(self.conv3(x)))
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn3(self.conv5(F.leaky_relu(self.conv4(x)))))
        x = F.leaky_relu(self.bn4(self.conv6(x)))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = EnhancedNet()

# Optimizer and Learning Rate Scheduler
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training the Network
for epoch in range(15):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Average Loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    scheduler.step()
print('Finished Training')

# Testing the Network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.3f} %')

# Save the Network
torch.save(net.state_dict(), './mnist_net.pth')
