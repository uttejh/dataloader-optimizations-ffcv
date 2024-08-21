import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import psutil
import matplotlib.pyplot as plt


def get_model():
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    return model


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
num_epochs = 20

train_set = torchvision.datasets.CIFAR10(root='/tmp', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='/tmp', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

net = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

total_time = 0.0

for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start = time.time()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print(f'Epoch {epoch + 1} took {epoch_time:.2f}s | Loss: {(running_loss / len(train_loader)):.3f}')

    total_time += epoch_time

print(f'Averaged {total_time / num_epochs:.2f}s per epoch')
