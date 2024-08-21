This repo is a playground to test and learn about the [FFCV library](https://github.com/libffcv/ffcv/tree/main) 
to increase data throughput while training with large image datasets.

### Comparison

Loading images with a pytorch dataloader took 1.7 seconds per epoch for CIFAR10 dataset with a batch size of 64. 
Using the FFCV library, the time was reduced to 0.18 seconds per epoch.

In regards to speed with augmentations (Horizontal flip), the time taken per epoch was 2.55 seconds
with pytorch dataloader while with FFCV, it was 0.33 seconds. However, there is limited support for 
[transforms](https://docs.ffcv.io/api/transforms.html) from ffcv.

### TODO
- [ ] Try with a 100 GB dataset