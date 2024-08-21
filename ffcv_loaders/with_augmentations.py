import time
from typing import List
import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage, RandomHorizontalFlip
from ffcv.transforms.common import Squeeze

batch_size = 64
num_epochs = 20

label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
image_pipeline: List[Operation] = [
    SimpleRGBImageDecoder(),
    RandomHorizontalFlip(),

    ToTensor(),
    ToDevice(torch.device('cuda:0')),
    ToTorchImage(),
    Convert(torch.float16),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

loader = Loader(f'/tmp/train.beton',
                batch_size=batch_size,
                num_workers=2,
                order=OrderOption.RANDOM,
                drop_last=True,
                pipelines={'image': image_pipeline,
                           'label': label_pipeline})

total_time = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start = time.time()
    for inputs, labels in loader:
        pass
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    print(f'Epoch {epoch + 1} took {epoch_time:.2f}s | Loss: {(running_loss / len(loader)):.3f}')

    total_time += epoch_time

print(f'Averaged {total_time / num_epochs:.2f}s per epoch')
