from utils import misc
import os
from utils import dataset_loader
from IPython import embed

known_models = [
    'alexnet', # 224x224
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', # 224x224
    'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152', # 224x224
    'squeezenet_v0', 'squeezenet_v1', # 224x224
    'inception_v3', # 299x299
    'mobilenet_v2', # 224x224
    'alexnet_cifar10', # 32x32
    'alexnet_cifar100', # 32x32
]

def alexnet(cuda=True, model_root=None):
    print("Building and initializing alexnet parameters")
    from models import alexnet as alx
    m = alx.alexnet(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def alexnet_cifar10(cuda=True, model_root=None):
    print("Building and initializing alexnet for Cifar 10 parameters")
    from models import alexnet_cifar as alx
    m = alx.alexnet_cifar10(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get10, False

def alexnet_cifar100(cuda=True, model_root=None):
    print("Building and initializing alexnet for Cifar 100 parameters")
    from models import alexnet_cifar as alx
    m = alx.alexnet_cifar100(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get100, False

def vgg16(cuda=True, model_root=None):
    print("Building and initializing vgg16 parameters")
    from models import vgg
    m = vgg.vgg16(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def vgg16_bn(cuda=True, model_root=None):
    print("Building vgg16_bn parameters")
    from models import vgg
    m = vgg.vgg19_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def vgg19(cuda=True, model_root=None):
    print("Building and initializing vgg19 parameters")
    from models import vgg
    m = vgg.vgg19(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def vgg19_bn(cuda=True, model_root=None):
    print("Building vgg19_bn parameters")
    from models import vgg
    m = vgg.vgg19_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def inception_v3(cuda=True, model_root=None):
    print("Building and initializing inception_v3 parameters")
    from models import inception
    m = inception.inception_v3(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def resnet18(cuda=True, model_root=None):
    print("Building and initializing resnet-18 parameters")
    from models import resnet
    m = resnet.resnet18(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def resnet34(cuda=True, model_root=None):
    print("Building and initializing resnet-34 parameters")
    from models import resnet
    m = resnet.resnet34(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def resnet50(cuda=True, model_root=None):
    print("Building and initializing resnet-50 parameters")
    from models import resnet
    m = resnet.resnet50(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def resnet101(cuda=True, model_root=None):
    print("Building and initializing resnet-101 parameters")
    from models import resnet
    m = resnet.resnet101(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def resnet152(cuda=True, model_root=None):
    print("Building and initializing resnet-152 parameters")
    from models import resnet
    m = resnet.resnet152(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def squeezenet_v0(cuda=True, model_root=None):
    print("Building and initializing squeezenet_v0 parameters")
    from models import squeezenet
    m = squeezenet.squeezenet1_0(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def squeezenet_v1(cuda=True, model_root=None):
    print("Building and initializing squeezenet_v1 parameters")
    from models import squeezenet
    m = squeezenet.squeezenet1_1(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def mobilenet_v2(cuda=True, model_root=None):
    print("Building and initializing mobilenet_v2 parameters")
    from models import mobilenet
    m = mobilenet.mobilenet_v2(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset_loader.get, True

def select(model_name, **kwargs):
    assert model_name in known_models, f'Supported models are {known_models}'
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return eval('{}'.format(model_name))(**kwargs)

if __name__ == '__main__':
    m1 = alexnet()
    embed()


