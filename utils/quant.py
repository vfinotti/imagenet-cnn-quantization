from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
from IPython import embed

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def get_scalling_factor(input, overflow_rate):
    """Calculates the scaling factor (sf) that better represents the input"""
    # transform 'input' into an array of the abs of each elements
    abs_value = input.abs().view(-1)
    # sort the modulus array in descending order
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    # find what index corresponds to the max possibe modulus value, considering the overflow_rate.
    # for '0' overflow_rate, the index will be the one of the maximum module of all modules, and
    # the biggest modulus (index 0) will be chosen
    split_idx = int(overflow_rate * len(sorted_value))
    # value at that index
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
    # get the minimum ammount of bits required to represent the value chosen and consider it the
    # scaling factor. The '1e-12' is there to determine the smallest precision (if 'v' is too small)
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def linear_quantize(input, sf, bits, return_type='float'):
    """Converts a float value from the real numbers domain to a float/int in the quantized domain"""
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1

    # calculates min and maximum. Half of the possible quantized space is for positive number and half
    # is for the negatives (that's why bound is 2^bits/2, or 2^(bits-1)). For 8 bits, the quantized
    # number will be between [-128,127].
    bound = math.pow(2.0, bits)/2 # half for positive, half for negative
    min_val = - bound
    max_val = bound - 1

    # We have "bits-1" to represent the positive numbers, and "bits-1" for the negative ones. Each of these
    # intervals are bound to represent the scalling factor that better represents the maximum modulus
    # of the input (2^sf). Therefore, we have "bits-1" bits to quantize the "2^sf" interval, and the
    # quantized input is a consequence of that.
    input.mul_(math.pow(2, bits-1)/math.pow(2,sf))
    input.float().round_()

    # the quantized input should be limited by the maximum and minimum values according to our design
    input.clamp_(min_val, max_val)

    # calculate the output format base on the return type desired
    assert return_type in ['float', 'int'], 'Return type should be \'float\' or \'int\'!'
    if return_type == 'float':
        # do the inverse operation, to return to the real domain number corresponding to
        # the quantization level
        input.clamp_(min_val, max_val)
        input.mul_(math.pow(2,sf)/math.pow(2, bits-1))
    elif return_type == 'int':
        input.clamp_(min_val, max_val)
    else:
        # format not supported, returning float
        input.clamp_(min_val, max_val)
        input.mul_(math.pow(2,sf)/math.pow(2, bits-1))

    return input

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits-1)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits-1)
    v = torch.exp(v) * s
    return v

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy()[0])
        min_val = float(min_val.data.cpu().numpy()[0])

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            # sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            # self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            sf_new = get_scalling_factor(input, self.overflow_rate)
            # get the biggest scalling factor, so no data is missed
            self.sf = max(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)

class Conv2dQuant(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bits, sf=None, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', overflow_rate=0.0, counter=10):

        from torch._six import container_abcs
        from itertools import repeat

        def _ntuple(n):
            def parse(x):
                if isinstance(x, container_abcs.Iterable):
                    return x
                return tuple(repeat(x, n))
            return parse

        _single = _ntuple(1)
        _pair = _ntuple(2)

        super(Conv2dQuant, self).__init__()
        self._counter = counter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sf = sf
        self.bits = bits
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.overflow_rate=overflow_rate
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *_pair(kernel_size)))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def conv2d_quant_forward(self, input, weight, bias, stride, padding, dilation, groups, sf, bits):
        if self.padding_mode == 'circular':
            raise NameError('Conv2dQuant does not support circular padding')

        in_samples, in_channels, in_height, in_width = input.shape
        conv2d_filter_num, _, kernel_height, kernel_width = weight.shape

        stride_height = stride[0]
        stride_width = stride[1]
        padding_height = padding[0]
        padding_width = padding[1]

        out_height = int((in_height - kernel_height + 2*padding_height)/stride_height) + 1
        out_width = int((in_width - kernel_width + 2*padding_width)/stride_width) + 1

        input_step = int(in_channels/groups)
        weight_step = int(conv2d_filter_num/groups)
        for i in range(groups):
            input_group = input[:,i*input_step:(i+1)*input_step,:,:]
            weight_group = weight[i*weight_step:(i+1)*weight_step,:,:,:]

            inp_unf = torch.nn.functional.unfold(input_group, (kernel_height, kernel_width), dilation=dilation, padding=padding, stride=stride)

            weight_t = weight_group.view(weight_group.size(0), -1).t()
            inp_unf_t = inp_unf.transpose(1, 2)
            inp_unf_t = inp_unf_t[:,:,:, None]
            inp_unf_t_exp = inp_unf_t.transpose(2,3).repeat((1,1,weight_t.t().size(0),1))
            weight_t_exp = weight_t.t().repeat((1,inp_unf_t.size(1),1,1))
            inp_unf_t_exp.mul_(weight_t_exp)
            out_unf = linear_quantize(inp_unf_t_exp, sf, bits)
            out_unf = linear_quantize(torch.sum(out_unf, dim=3).transpose(1, 2), sf, bits)
            if bias is not None:
                out_unf = linear_quantize(out_unf + bias.view(-1, 1), sf, bits)

            if i==0:
                out = out_unf.view(in_samples, weight_step, out_height, out_width)
            else:
                out = torch.cat((out,out_unf.view(in_samples, weight_step, out_height, out_width)), dim=1)

        return out

    def conv2d_forward(self, input, weight, bias, stride, padding, dilation, groups):
        if self.padding_mode == 'circular':
            raise NameError('Conv2d does not support circular padding')

        in_samples, in_channels, in_height, in_width = input.shape
        conv2d_filter_num, _, kernel_height, kernel_width = weight.shape

        stride_height = stride[0]
        stride_width = stride[1]
        padding_height = padding[0]
        padding_width = padding[1]

        out_height = int((in_height - kernel_height + 2*padding_height)/stride_height) + 1
        out_width = int((in_width - kernel_width + 2*padding_width)/stride_width) + 1

        input_step = int(in_channels/groups)
        weight_step = int(conv2d_filter_num/groups)
        for i in range(groups):
            input_group = input[:,i*input_step:(i+1)*input_step,:,:]
            weight_group = weight[i*weight_step:(i+1)*weight_step,:,:,:]
            inp_unf = torch.nn.functional.unfold(input_group, (kernel_height, kernel_width), dilation=dilation, padding=padding, stride=stride)

            if bias is None:
                out_unf = (inp_unf.transpose(1, 2).matmul(weight_group.view(weight_group.size(0), -1).t())).transpose(1, 2)
            else:
                out_unf = (inp_unf.transpose(1, 2).matmul(weight_group.view(weight_group.size(0), -1).t()) + bias).transpose(1, 2)

            if i==0:
                out = out_unf.view(in_samples, weight_step, out_height, out_width)
            else:
                out = torch.cat((out,out_unf.view(in_samples, weight_step, out_height, out_width)), dim=1)

        return out

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            output = self.conv2d_forward(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            sf_new = get_scalling_factor(output, self.overflow_rate)
            # get the biggest scalling factor, so no data is missed
            self.sf = max(self.sf, sf_new) if self.sf is not None else sf_new
            return output
        else:
            output = self.conv2d_quant_forward(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}({}, {}, kernel_size={}, sf={}, bits={}, stride={}, padding={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.kernel_size, self.sf, self.bits, self.stride, self.padding)

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    # quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model

