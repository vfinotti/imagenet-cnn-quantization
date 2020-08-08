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
  - AlexNet for Cifar10 and Cifar100
  - SqueezeNet\_v1 for Cifar10 and Cifar100
  - MobileNet_v2 for Cifar10 and Cifar100
  - ResNet18 for Cifar10 and Cifar100
  - ResNet50 for Cifar10 and Cifar100

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
| &#x2011;&#x2011;type                   |Argument| squeezenet_v1                  | Selected model for inference. <br>Available values: `alexnet`, `vgg16`, `vgg19`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `squeezenet_v0`, `squeezenet_v1`, `alexnet_cifar10`, `alexnet_cifar100`, `squeezenet_v1_cifar10`, `squeezenet_v1_cifar100`, `mobilenet_cifar10`, `mobilenet_cifar100`, `resnet18_cifar10`, `resnet18_cifar100`, `resnet50_cifar10`, `resnet50_cifar100` |
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

We evaluated the impact of quantization on a few popular datasets and architectures. For the purpose of avoiding long execution time, only the first 1000 samples were used when computing network accuracy.

| Model                                                                                                                             | 8-bits       | 10-bits      | 12-bits      | 16-bits      | floating-point (32-bits)  |
| :-------------------------------------------------------------------------------------------------------------------------------- | :----------: | :----------: | :----------: | :----------: | :-----------------------: |
| [AlexNet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)<sup>1</sup>                                               | 00.10/00.90  | 24.48/47.55  | 53.25/77.12  | 55.04/79.02  | 55.24/78.92               |
| [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)<sup>1</sup>                                                 | 00.70/01.90  | 54.85/79.82  | 69.93/88.91  | 69.03/89.91  | 69.03/89.71               |
| [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)<sup>1</sup>                                                 | 00.50/02.20  | 56.44/78.72  | 75.42/92.31  | 75.12/93.11  | 75.12/93.11               |
| [SqueezeNet v1](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)<sup>1</sup>                                       | 01.10/04.40  | 52.95/75.02  | 57.74/79.52  | 58.44/79.52  | 58.24/79.72               |
| [MobileNet v2](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)<sup>1</sup>                                        | 00.10/00.70  | 08.29/21.28  | 67.63/88.81  | 71.83/89.71  | 71.63/89.71               |
| [AlexNet for Cifar10](https://dl.dropboxusercontent.com/s/vc3l9h74lxc1udl/alexnet-cifar10-edd17383.pth)<sup>2</sup>               | 21.88/67.63  | 80.22/99.10  | 92.21/100.0  | 92.61/100.0  | 92.51/100.0               |
| [AlexNet for Cifar100](https://dl.dropboxusercontent.com/s/k5euygnxshnkeni/alexnet-cifar100-2822cb79.pth)<sup>2</sup>             | 03.80/13.29  | 53.05/80.72  | 69.53/90.71  | 70.13/90.51  | 70.03/90.51               |
| [SqueezeNet v1 for Cifar10](https://dl.dropboxusercontent.com/s/ftipf6s67fwbv0o/squeezenet1_1-cifar10-6d314894.pth)<sup>2</sup>   | 23.78/72.33  | 88.01/99.90  | 91.41/99.90  | 91.21/99.90  | 91.11/99.90               |
| [SqueezeNet v1 for Cifar100](https://dl.dropboxusercontent.com/s/zr1bff1d69b40pa/squeezenet1_1-cifar100-0676d27d.pth)<sup>2</sup> | 16.78/36.96  | 67.03/89.91  | 68.93/91.11  | 68.73/91.41  | 68.83/91.41               |
| [MobileNet v2 for Cifar10](https://dl.dropboxusercontent.com/s/jp7ywvqyl8jdyd2/mobilenet-cifar10-6ad0292d.pth)<sup>2</sup>        | 10.19/50.15  | 19.58/61.04  | 94.01/99.90  | 95.30/99.90  | 95.40/99.90               |
| [MobileNet v2 for Cifar100](https://dl.dropboxusercontent.com/s/8ai35hn6vjvhh8b/mobilenet-cifar100-27192b0e.pth)<sup>2</sup>      | 00.70/04.40  | 07.49/20.98  | 74.93/93.91  | 78.52/94.41  | 78.42/94.41               |
| [ResNet18 for Cifar10](https://dl.dropboxusercontent.com/s/5q9wq89360e1suz/resnet18-cifar10-a454c9e8.pth)<sup>2</sup>             | 10.79/51.45  | 88.21/99.30  | 95.60/100.0  | 96.00/100.0  | 95.90/100.0               |
| [ResNet18 for Cifar100](https://dl.dropboxusercontent.com/s/2zzvqfeda9pdyeu/resnet18-cifar100-45405922.pth)<sup>2</sup>           | 03.10/10.59  | 69.13/89.41  | 79.82/94.71  | 80.12/95.50  | 80.22/95.40               |
| [ResNet50 for Cifar10](https://dl.dropboxusercontent.com/s/49s5tdclxilaxbv/resnet50-cifar10-895fa48d.pth)<sup>2</sup>             | 11.59/49.15  | 81.12/99.70  | 96.10/100.0  | 96.50/100.0  | 96.50/100.0               |
| [ResNet50 for Cifar100](https://dl.dropboxusercontent.com/s/5fii6ixf1la8xwf/resnet50-cifar100-525d0dbc.pth)<sup>2</sup>           | 01.40/05.79  | 52.65/76.92  | 83.02/96.00  | 84.42/96.00  | 84.32/96.00               |

**<sup>1</sup>ImageNet 32-float models are directly from torchvision**<br>
**<sup>2</sup>Retrained with transfer-learning for 30 epochs, changing the number of output classes to 10 (Cifar10) or 100 (Cifar100)**

## Acknowledgements

The repository structure and code was based on the incredible work done by
[https://github.com/aaron-xichen/pytorch-playground/blob/master/README.md](https://github.com/aaron-xichen/pytorch-playground/blob/master/README.md).
