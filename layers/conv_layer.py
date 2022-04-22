from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch
import math


class CustomConv(nn.Module):
    """Shamelessly stolen from https://github.com/pytorch/pytorch/issues/47990
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(CustomConv, self).__init__()
        self.k = self._to_tuple(kernel_size)
        self.in_c = in_channels
        self.out_c = out_channels
        self.stride = self._to_tuple(stride)
        if padding == 'same':
            if stride != 1 and stride != (1,1):
                raise Exception("Padding 'same' can only be used with stride = 1")
            self.padding = padding
        else:
            self.padding = self._to_tuple(padding)

        self.weight = nn.Parameter(torch.ones((out_channels, in_channels, self.k[0], self.k[1]), dtype=torch.float64), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_channels, dtype=torch.float64), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def _to_tuple(self, x):
        if isinstance(x, tuple):
            return x
        return x, x

    def forward(self, x):
        k = self.k
        stride = self.stride
        h_in, w_in = x.shape[2], x.shape[3]

        padding = self.padding
        h_out, w_out = None, None
        if padding == 'same':
            padding_h = int(((h_in - 1) - h_in + (k[0]-1) + 1)/2)
            padding_w = int(((w_in - 1) - w_in + (k[1]-1) + 1)/2)
            padding = (padding_h, padding_w)
        h_out = (h_in + 2 * padding[0] - (k[0] - 1) - 1) / stride[0] + 1
        w_out = (w_in + 2 * padding[1] - (k[1] - 1) - 1) / stride[1] + 1
        h_out, w_out = int(h_out), int(w_out)

        x_unf = F.unfold(x, k, padding=padding, stride=stride)
        out_unf = x_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        out_ = F.fold(out_unf, (h_out, w_out), (1, 1)) + self.bias.view(-1, 1, 1)
        return out_
