from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch
import math


class LocallyConnected(nn.Module):
    """Adapted from https://github.com/pytorch/pytorch/issues/47990
    """
    def __init__(self, in_shape, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(LocallyConnected, self).__init__()
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

        L = self._num_blocks(in_shape)
        K = self.k[0] * self.k[1] * in_channels
        self.weight = nn.Parameter(torch.ones((out_channels, L, K), dtype=torch.float64), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty((out_channels, L), dtype=torch.float64), requires_grad=True)
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

    def _set_padding(self, in_shape):
        h_in, w_in = in_shape
        padding_h = int(((h_in - 1) - h_in + (self.k[0]-1) + 1)/2)
        padding_w = int(((w_in - 1) - w_in + (self.k[1]-1) + 1)/2)
        return (padding_h, padding_w)

    def _num_blocks(self, in_shape):
        if len(in_shape) == 4:
            in_shape = (in_shape[2], in_shape[3])
        padding = self.padding
        if padding == 'same':
            padding = self._set_padding(in_shape)
        num_blocks = 1
        for d in [0,1]:
            numer = in_shape[d] + 2 * padding[d] - (self.k[d] - 1) - 1
            denom = self.stride[d]
            num_blocks *= int(np.floor(numer/denom + 1))
        return num_blocks

    def forward(self, x):
        k = self.k
        stride = self.stride
        h_in, w_in = x.shape[2], x.shape[3]

        padding = self.padding
        h_out, w_out = None, None
        if padding == 'same':
            padding = self._set_padding((h_in, w_in))
        h_out = (h_in + 2 * padding[0] - (k[0] - 1) - 1) / stride[0] + 1
        w_out = (w_in + 2 * padding[1] - (k[1] - 1) - 1) / stride[1] + 1
        h_out, w_out = int(h_out), int(w_out)

        x_unf = F.unfold(x, k, padding=padding, stride=stride).transpose(1, 2)
        N, L, K = x_unf.shape
        out_unf = (x_unf.view((N, 1, L, K)) * self.weight).sum(-1)
        out_f = F.fold(out_unf, (h_out, w_out), (1, 1))
        if self.bias is not None:
            out_f = out_f + self.bias.view(-1, out_f.shape[2], out_f.shape[3])
        return out_f