# Simulating quantization on ImageNet CNNs

This repository contains the code to simulate quantization effects on several ImageNet networks using Pytorch. Through the quantization of weights, bias and the replacement of conventional Pytorch layers by custom quantized implementations of them (which enforce the rounding behaviour of fixed-point operations after every arithmetical operation), we manage to simulate quantized inference on several existing CNN's without the need of implementing fixed-point arithmetic. 

## Main features

- Weights and bias quantization
- Simulation of signed two's complement fixed-point quantization during network inference
- GPU or CPU-only modes
- Result reports are saved on a separate _acc1\_acc5.txt_ file
- Supported ImageNet architectures
  - AlexNet
  - VGG16, VGG19
  - ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
  - SqueezeNet\_v0, SqueezeNet\_v1
  - MobileNet_v2 
- Other supported datasets and architectures
  - AlexNet for Cifar10, AlexNet for Cifar100

## Getting started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. 

### Prerequisites

- Python 3+
- Conda
- Linux or Windows

### Installing
In order to install the environment with all the require dependencies, create the environment "_pytorch-env_" by running the following commands on your terminal:

If you are running a *Linux* machine:

```sh
$ conda env create -f environment-linux.yml
```

If you are running a *Windows* machine:

```sh
$ conda env create -f environment-win.yml
```

Activate the environment:

```sh
$ conda activate pytorch-env
```

### ImageNet dataset installation

Since ImageNet dataset is not open for public, we are using a pre-computed imagenet validation dataset with 224x224x3 size provided by [https://github.com/aaron-xichen/pytorch-playground](https://github.com/aaron-xichen/pytorch-playground/issues/18). The images were first resized to 256, then cropped on a 224x224 area around the center. The resulting images were encoded as a _JPEG_ string and dumped to pickle. 

In order to download it:

- `cd script`
- Download the _val224\_compressed.pkl_ file ([Tsinghua](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl) /  [Google Drive](https://drive.google.com/file/d/1U8ir2fOR4Sir3FCj9b7FQRPSVsycTfVc/view?usp=sharing)) to that directory
- run `python extract_imagenet_dataset.py` (needs 16G of RAM)

## Usage

Run the simulated quantized inference by executing _quantize.py_ with the desired arguments or flags:

| Name                                   | Type   | Default value                  | Description|
|:---------------------------------------|--------|:------------------------------:|:-----------|
| &#x2011;&#x2011;type                   |Argument| squeezenet_v1                  | Selected model for inference. <br>Available values: `alexnet`, `vgg16`, `vgg19`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `squeezenet_v0`, `squeezenet_v1`, `alexnet_cifar10`, `alexnet_cifar100` |
| &#x2011;&#x2011;batch&#x2011;size      |Argument| 1                              | Input batch size for training|
| &#x2011;&#x2011;param&#x2011;bits      |Argument| 16                             | Bit-width of weights and bias|
| &#x2011;&#x2011;fwd&#x2011;bits        |Argument| 16                             | Bit-width of activation|
| &#x2011;&#x2011;n&#x2011;samples       |Argument| 20                             | Number of samples taken to define quantization parameters before inference|
| &#x2011;&#x2011;use&#x2011;cpu         |Flag    | False                          | Enforce CPU-only execution execution|

### Examples

- AlexNet for Cifar10, 13 bits for weight/bias quantization and 13 bits for activations. Using GPU (which is default):
```sh
$ python quantize.py --type alexnet_cifar10 --param-bits 13 --fwd-bits 13
```

- SqueezeNet, 11 bits for weight/bias quantization and 11 bits for activations. Using GPU (which is default):

```sh
$ python quantize.py --type squeezenet_v1 --param-bits 11 --fwd-bits 11
```
- AlexNet, 16 bits for weight/bias quantization and 16 bits for activations. Using only CPU:

```sh
$ python quantize.py --type alexnet --param-bits 16 --fwd-bits 16 --use-cpu
```

## Top1/Top5 accuracy results

We evaluated the impact of quantization on a few popular datasets and architectures. For the purpose of avoiding long execution time, only the first 1000 samples were used for ImageNet architectures when computing network accuracy. For Cifar10 and Cifar100, all the 10000 samples were used.

|Model                                                                                                                 |8-bits      |10-bits     |12-bits     |16-bits     |floating-point (32-bits) |
|:---------------------------------------------------------------------------------------------------------------------|:----------:|:----------:|:----------:|:----------:|:-----------------------:|
|[AlexNet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)<sup>1</sup>                                   |00.10/00.90 |24.48/47.55 |53.25/77.12 |55.04/79.02 |55.24/78.92|
|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)<sup>1</sup>                                     |00.70/01.90 |54.85/79.82 |69.93/88.91 |69.03/89.91 |69.03/89.71|
|[ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)<sup>1</sup>                                     |00.50/02.20 |56.44/78.72 |75.42/92.31 |75.12/93.11 |75.12/93.11|
|[SqueezeNet v1](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)<sup>1</sup>                           |01.10/04.40 |52.95/75.02 |57.74/79.52 |58.44/79.52 |58.24/79.72|
|[MobileNet v2](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)<sup>1</sup>                            |00.10/00.70 |08.29/21.28 |67.63/88.81 |71.83/89.71 |71.63/89.71|
|[AlexNet for Cifar10](https://dl.dropboxusercontent.com/s/5wyve2lob61o0xz/alexnet-cifar10-cbfee098.pth)<sup>2</sup>   |26.21/80.09 |47.39/89.18 |49.19/89.43 |49.32/89.52 |49.34/89.47|
|[AlexNet for Cifar100](https://dl.dropboxusercontent.com/s/towf68tohx5uv5q/alexnet-cifar100-c43d2c5d.pth)<sup>2</sup> |07.07/21.11 |28.09/55.44 |30.34/57.92 |30.83/57.95 |30.76/57.98|

**<sup>1</sup>ImageNet 32-float models are directly from torchvision**<br>
**<sup>2</sup>AlexNet architecture adaptation for Cifar10 and Cifar100 was obtained from [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/)**

## Acknowledgements

The repository structure and code was based on the incredible work done by
[https://github.com/aaron-xichen/pytorch-playground/blob/master/README.md](https://github.com/aaron-xichen/pytorch-playground/blob/master/README.md).

For the classification of Cifar10 and Cifar100, we used the AlexNet-based architecture obtained from [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/).
