from utils import misc
import os
import os.path
import numpy as np
import joblib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get(batch_size, data_root='./datasets', train=False, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'imagenet-data'))
    print("Building IMAGENET data loader, 50000 for train, 50000 for test")
    ds = []
    assert train is not True, 'train not supported yet'
    if train:
        ds.append(IMAGENET(data_root, batch_size, True, **kwargs))
    if val:
        ds.append(IMAGENET(data_root, batch_size, False, **kwargs))
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get10(batch_size, data_root='./datasets', train=False, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    input_size = 224
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get100(batch_size, data_root='./datasets', train=False, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    input_size = 224
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class IMAGENET(object):
    def __init__(self, root, batch_size, train=False, input_size=224, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.train = train

        if train:
            pkl_file = os.path.join(root, 'train{}.pkl'.format(input_size))
        else:
            pkl_file = os.path.join(root, 'val{}.pkl'.format(input_size))
        self.data_dict = joblib.load(pkl_file)

        self.batch_size = batch_size
        self.idx = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample* 1.0 / self.batch_size))

    @property
    def n_sample(self):
        return len(self.data_dict['data'])

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_batch:
            self.idx = 0
            raise StopIteration
        else:
            img = self.data_dict['data'][self.idx*self.batch_size:(self.idx+1)*self.batch_size].astype('float32')
            target = self.data_dict['target'][self.idx*self.batch_size:(self.idx+1)*self.batch_size]
            self.idx += 1
            return img, target

if __name__ == '__main__':
    train_ds, val_ds = get(200)


