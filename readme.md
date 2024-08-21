This repo is a playground to test and learn about the [FFCV library](https://github.com/libffcv/ffcv/tree/main) 
to increase data throughput while training with large image datasets.

### Comparison

Loading images with a pytorch dataloader took 1.7 seconds per epoch for CIFAR10 dataset with a batch size of 64. 
Using the FFCV library, the time was reduced to 0.18 seconds per epoch.