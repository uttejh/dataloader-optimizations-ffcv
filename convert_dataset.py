import torchvision
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


datasets = {
    'train': torchvision.datasets.CIFAR10('/tmp', train=True, download=True),
    'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
    }

for (name, ds) in datasets.items():
    writer = DatasetWriter(f"/tmp/{name}.beton", {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

