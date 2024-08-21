import time
from typing import List
import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torchvision.models import resnet18


def get_model():
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    return model


batch_size = 64
num_epochs = 20

label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
image_pipeline: List[Operation] = [
    SimpleRGBImageDecoder(),
    ToTensor(),
    ToDevice(torch.device('cuda:0')),
    ToTorchImage(),
    Convert(torch.float16),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

loader = Loader(f'/tmp/train.beton',
                       batch_size=batch_size,
                       num_workers=2,
                       order=OrderOption.RANDOM,
                       drop_last=True,
                       pipelines={'image': image_pipeline,
                                  'label': label_pipeline})

net = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

total_time = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start = time.time()
    for inputs, labels in loader:
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print(f'Epoch {epoch + 1} took {epoch_time:.2f}s | Loss: {(running_loss / len(loader)):.3f}')

    total_time += epoch_time

print(f'Averaged {total_time / num_epochs:.2f}s per epoch')
