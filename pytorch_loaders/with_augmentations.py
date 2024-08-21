import time
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
num_epochs = 20

train_set = torchvision.datasets.CIFAR10(root='/tmp', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

total_time = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start = time.time()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print(f'Epoch {epoch + 1} took {epoch_time:.2f}s | Loss: {(running_loss / len(train_loader)):.3f}')

    total_time += epoch_time

print(f'Averaged {total_time / num_epochs:.2f}s per epoch')
