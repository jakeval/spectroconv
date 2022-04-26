from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch
import math
from layers import local_layer


class LocallyConnectedCombiningWeights(nn.Module):
    def __init__(self, rank, output_shape, in_channels, low_dim_channels, compression=1):
        super(LocallyConnectedCombiningWeights, self).__init__()
        self.size = output_shape
        compression = (compression, compression)

        self.dim_reduction_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=low_dim_channels,
            kernel_size=1,
            stride=compression,
            bias=True)

        self.lc_layer = local_layer.DepthwiseLocallyConnected(
            output_shape,
            low_dim_channels,
            kernel_size=1,
            stride=compression,
            bias=True
        )

        shape = self.downsample_shape(output_shape, compression)
        padding = self.get_padding(shape, output_shape, compression)

        self.proj_layer = nn.ConvTranspose2d(
            low_dim_channels,
            out_channels=rank,
            kernel_size=1,
            stride=compression,
            output_padding=padding,
            bias=True
        )

    def forward(self, x):
        x_lowd = self.dim_reduction_layer(x)
        x_lowd = F.interpolate(x_lowd, self.size, mode='bilinear', align_corners=True)
        x_lc = self.lc_layer(x_lowd)
        output = self.proj_layer(x_lc)
        return F.softmax(output, dim=-1)

    def downsample_shape(self, input_shape, stride):
        shape = [None, None]
        for d in range(2):
            shape[d] = int(math.floor((input_shape[d] - 1) / stride[d] + 1))
        return tuple(shape)

    def get_padding(self, input_shape, output_shape, stride):
        p = [None, None]
        for d in range(2):
            p[d] = int(output_shape[d] - (input_shape[d] - 1) * stride[d] - 1)
        return tuple(p)


class LocalizationCombiningWeights(nn.Module):
    def __init__(self, rank, input_shape, output_shape, in_channels, low_dim_channels, local_dim=None, aggregation='max'):
        super(LocalizationCombiningWeights, self).__init__()
        self.size = output_shape
        self.local_dim = local_dim
        self.aggregation = aggregation

        self.dim_reduction_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=low_dim_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        
        input_size = min(input_shape)
        self.dilations = [1,2,4,8]
        self.multiscale_layers = []

        k = 3
        for r in self.dilations:
            #dilation_size = (k + (r-1)*(k-1))
            if r <= int((input_size - 1) / 2):
                self.multiscale_layers.append(
                    nn.Conv2d(
                        in_channels=low_dim_channels,
                        out_channels=low_dim_channels,
                        kernel_size=k,
                        stride=1,
                        padding='valid',
                        bias=True,
                        groups=low_dim_channels,
                        dilation=r))

        multiscale_out = (len(self.multiscale_layers) + 2) * low_dim_channels
        ch1 = (len(self.multiscale_layers) + 2) * rank
        ch2 = int(ch1 // 2)
        
        self.squeeze_layer = nn.Conv2d(
            in_channels=multiscale_out,
            out_channels=ch2,
            kernel_size=1,
            stride=1,
            bias=True)

        self.excite_layer = nn.Conv2d(
            in_channels=ch2,
            out_channels=ch1,
            kernel_size=1,
            stride=1,
            bias=True)

        self.proj_layer = nn.Conv2d(
            in_channels=ch1,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            bias=True)

    def forward(self, x):
        x_lowd = self.dim_reduction_layer(x)
        x_pool = torch.mean(x_lowd, dim=(2,3), keepdim=True)

        x_multiscale = [
            F.interpolate(x_lowd, self.size, mode='bilinear', align_corners=True),
            F.interpolate(x_pool, self.size, mode='bilinear', align_corners=True)
        ]

        for layer in self.multiscale_layers:
            x_multiscale.append(
                F.interpolate(layer(x_lowd), self.size, mode='bilinear', align_corners=True))
        x_multiscale = torch.concat(x_multiscale, dim=1)

        if self.local_dim is not None:
            d = 3 - self.local_dim # average over shared dimension
            if self.aggregation == 'max':
                x_multiscale = x_multiscale.max(dim=d, keepdim=True).values
            if self.aggregation == 'mean':
                x_multiscale = x_multiscale.mean(dim=d, keepdim=True)

        x_s = self.squeeze_layer(x_multiscale)
        x_s = F.relu(x_s)
        x_e = self.excite_layer(x_s)
        x_e = torch.sigmoid(x_e)
        output = self.proj_layer(x_e)
        output = F.softmax(output, dim=1)
        return output

    def _random_init(self):
        pass


class OuterCombiningWeights(nn.Module):
    def __init__(self, rank, output_shape, local_dim=None):
        super(OuterCombiningWeights, self).__init__()
        self.rank = rank
        self.output_shape = output_shape
        self.local_dim = local_dim
        self.init_combining_weights(self.output_shape, self.rank)

    def init_combining_weights(self, L, R):
        c = 1 / 2
        L_h, L_w = L
        if self.local_dim is None or self.local_dim == 0:
            self.wts_h = Parameter(torch.ones((R, L_h, 1)) * c/(np.sqrt(R)), requires_grad=True)
        else:
            self.register_parameter('wts_h', None)
        if self.local_dim is None or self.local_dim == 1:
            self.wts_w = Parameter(torch.ones((R, 1, L_w)) * c/(np.sqrt(R)), requires_grad=True)
        else:
            self.register_parameter('wts_w', None)

    def _random_init(self):
        L_h, L_w = self.output_shape
        with torch.no_grad():
            if self.local_dim is None or self.local_dim == 0:
                self.wts_h = Parameter(torch.rand((self.rank, L_h, 1)), requires_grad=True)
            if self.local_dim is None or self.local_dim == 1:
                self.wts_w = Parameter(torch.rand((self.rank, 1, L_w)), requires_grad=True)

    def forward(self):
        """
        outputs:
        d=None: 1, R, Lh, Lw
        d=0: 1, R, Lh, 1
        d=1: 1, R, 1, Lw
        """
        wts_h, wts_w = 0, 0
        if self.local_dim is None or self.local_dim == 0:
            wts_h = self.wts_h
        if self.local_dim is None or self.local_dim == 1:
            wts_w = self.wts_w
        return torch.softmax(wts_w + wts_h, dim=0)[None,:]


class LowRankLocallyConnected(nn.Module):
    """Adapted from https://github.com/google-research/google-research/blob/master/low_rank_local_connectivity/layers.py"""
    def __init__(self, rank, in_shape, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, local_dim=None):
        super(LowRankLocallyConnected, self).__init__()
        self.rank = rank
        self.local_dim = local_dim
        self.k = self._to_tuple(kernel_size)
        self.in_c = in_channels
        self.out_c = out_channels
        self.stride = self._to_tuple(stride)
        self.use_bias = bias
        if padding == 'same':
            if stride != 1 and stride != (1,1):
                raise Exception("Padding 'same' can only be used with stride = 1")
            self.padding = padding
        else:
            self.padding = self._to_tuple(padding)
        self.L = self._num_blocks(in_shape)
        L_h, L_w = self.L

        # initialize bias
        if self.use_bias:
            self.init_bias(L_h, L_w, self.out_c)
        else:
            self.register_parameter('bias_h', None)
            self.register_parameter('bias_w', None)
            self.register_parameter('bias_c', None)
            
        self.init_weight_basis(self.k, self.rank, self.out_c, self.in_c)

    def init_bias(self, L_h, L_w, out_c):
        self.bias_c = Parameter(torch.zeros((out_c, 1, 1)))

        if self.local_dim is None or self.local_dim == 0:
            self.bias_h = Parameter(torch.zeros((1, L_h, 1)))
        else:
            self.register_parameter('bias_h', None)

        if self.local_dim is None or self.local_dim == 1:
            self.bias_w = Parameter(torch.zeros((1, 1, L_w)))
        else:
            self.register_parameter('bias_w', None)

    def init_weight_basis(self, k, R, out_channels, in_channels):
        """
        Basis shape: C2*R, C1, Kh, Kw
        """
        self.weight_bases = nn.Parameter(torch.ones((out_channels * R, in_channels, k[0], k[1]), dtype=torch.float64), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_bases, mode='fan_out', nonlinearity='relu')

    def get_weight(self, combining_weights, weight_bases=None, tile=False):
        """
        Used only for debugging and visualization.

        comining_weights: N, R, Lh, Lw
        weight_bases: C2*R, C1, Kh, Kw

        output: N, Lh, Lw, C2, C1, Kh, Kw
        """
        if weight_bases is None:
            weight_bases = self.weight_bases

        if tile and self.local_dim is not None:
            combining_weights = torch.tile(combining_weights, (1, 1, *self._tile()))

        return torch.tensordot(
            combining_weights,
            weight_bases.view(self.rank, self.out_c, self.in_c, self.k[0], self.k[1]),
            dims=([1], [0])).detach()

    def get_bias(self, tile=False):
        bias_h = 0
        bias_w = 0
        if self.local_dim is None or self.local_dim == 0:
            bias_h = self.bias_h
        if self.local_dim is None or self.local_dim == 1:
            bias_w = self.bias_w
        bias = bias_h + bias_w + self.bias_c
        if tile and self.local_dim is not None:
            bias = torch.tile(bias, (1, *self._tile()))
        return bias

    def forward(self, x, combining_weights):
        """
        x: N, C1, Hh, Hw
        combining_weights: N, R, Lh, Lw

        outputs: N, C2, Lh, Lw
        """
        convs = self.basis_convolve(x)
        outputs = torch.einsum('ijklm,ijlm->iklm', convs, combining_weights)
        if self.use_bias:
            outputs = outputs + self.get_bias()
        return outputs

    def basis_convolve(self, x):
        convs = F.conv2d(
            x,
            weight=self.weight_bases,
            stride=self.stride,
            padding=self.padding).view(
                x.shape[0],
                self.rank,
                self.out_c,
                self.L[0],
                self.L[1])
        return convs

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
        padding = self.padding
        if padding == 'same':
            padding = self._set_padding(in_shape)
        L = [None, None]
        for d in [0,1]:
            numer = in_shape[d] + 2 * padding[d] - (self.k[d] - 1) - 1
            denom = self.stride[d]
            L[d] = int(np.floor(numer/denom + 1))
        return tuple(L)

    def _tile(self):
        L = list(self.L)
        L[self.local_dim] = 1
        return tuple(L)
