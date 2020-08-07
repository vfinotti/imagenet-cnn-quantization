import torch
import torch.nn as nn
from .utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet_cifar10', 'alexnet_cifar100']


model_urls = {
    'alexnet_cifar10': 'https://dl.dropboxusercontent.com/s/vc3l9h74lxc1udl/alexnet-cifar10-edd17383.pth',
    'alexnet_cifar100': 'https://dl.dropboxusercontent.com/s/k5euygnxshnkeni/alexnet-cifar100-2822cb79.pth',
}

def fix_state_dict_keys(state_dict):
    '''Remove ".module" prefix from state_dict keys'''
    new_state_dict = state_dict.copy()
    for key in state_dict.keys():
        if key[0:7] == 'module.':
            new_state_dict[key[7:]] = state_dict[key]
            new_state_dict.pop(key)
    return new_state_dict


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet_cifar10(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(num_classes=10, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet_cifar10'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def alexnet_cifar100(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(num_classes=100, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet_cifar100'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
