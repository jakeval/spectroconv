import numpy as np
import torch
from torch import nn
from layers import lrlc_layer
import itertools
from models import lrlc_model


class LayerActivationMonitor:
    """Grabs the layer output from within a model"""
    def __init__(self, layer, detach=True):
        self.output = None
        self.detach = detach
        layer.register_forward_hook(self._hook)

    def get_layer_output(self):
        return self.output

    def _hook(self, model, input, output):
        if self.detach:
            self.output = output.detach()
        else:
            self.output = output


class BasisActivationMonitor:
    """Grabs the LRLC basis convolution output from within a model"""
    def __init__(self, layer, detach=True):
        self.output = None
        self.detach = detach
        self.layer = layer
        layer.register_forward_hook(self._hook)

    def get_layer_output(self):
        return self.output

    def _hook(self, model, input, output):
        x, cw = input
        output = self.layer.basis_convolve(x)
        if self.detach:
            self.output = output.detach()
        else:
            self.output = output


def receptive_field(x, out, model, selection):
    """Adapted from https://learnopencv.com/cnn-receptive-field-computation-using-backprop/.
    Uses backprop to find the receptive field of a neuron.
    
    x: N x C x H x W
    out: the output of the layer of interest
    model: the model of interest
    selection: the neuron in the layer of interest. Always prepended with (0,) for the N dimension
               CNN example: (0, 0, lh, lw)
               linear example: (0, neuron_id)
    """
    grad = torch.zeros_like(out, requires_grad=True)
    with torch.no_grad():
        idx = tuple([torch.LongTensor([v]) for v in list(selection)])
        grad.index_put_(idx, torch.tensor([1]).float())
    out.backward(gradient=grad, retain_graph=True)
    x_grad = x.grad[0, 0].data.numpy()
    x_grad = x_grad / np.amax(x_grad)
    model.zero_grad()
    x.grad = None
    return x_grad


def prepare_model(model):
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].detach().clone()
    model.train()
    #if isinstance(model, lrlc_model.LRLCTaenzer):
    #    b = model.lrlc
    #    model.lrlc = lrlc_model.TaenzerLRLCBlock(b.in_shape, b.rank, b.in_channels, b.out_channels, b.local_dim, False, b.low_dim, b.dropout_, b.device).float()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d) and module.affine:
            module.eval()
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
            nn.init.constant_(module.weight, 0.05)
            nn.init.zeros_(module.bias)
        elif isinstance(module, lrlc_layer.LowRankLocallyConnected):
            nn.init.constant_(module.weight_bases, 0.05)
            if module.bias_c is not None:
                nn.init.zeros_(module.bias_c)
            if module.bias_w is not None:
                nn.init.zeros_(module.bias_w)
            if module.bias_h is not None:
                nn.init.zeros_(module.bias_h)
        elif isinstance(module, lrlc_layer.OuterCombiningWeights):
            module.eval()
        elif isinstance(module, torch.nn.modules.Conv2d):
            nn.init.constant_(module.weight, 0.05)
            nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Linear):
            nn.init.constant_(module.weight, 0.05)
            nn.init.zeros_(module.bias)
        else:
            pass

    return model, state_dict


def restore_model(model, state_dict):
    model.load_state_dict(state_dict)
    return model


def layer_receptive_field(input_shape, output_dims, model, layer):
    """Returns the receptive field of each neuron in the layer
    
    input_shape: the model input shape. H x W
    output_dims: how many dimensions each neuron has. Linear has one dimension, CNN has two
    model: the model
    layer: the layer to select neurons from

    Returns a receptive field for every neuron in the output.
        The shape is [Neuron dims] x H x W
    """
    model, state_dict = prepare_model(model)

    x = torch.ones((1, 1, *input_shape), requires_grad=True)
    activation = LayerActivationMonitor(layer, detach=False)
    _ = model(x)
    out = activation.get_layer_output()
    output_shape = tuple(list(out.shape)[-output_dims:])
    ignored_dims = len(out.shape) - len(output_shape)
    ignored_dims = tuple([0]*ignored_dims)

    rfield = np.zeros((*output_shape, *input_shape))
    indices = itertools.product(*[list(range(dim)) for dim in output_shape])
    for idx in indices:
        rfield[idx] += receptive_field(x, out, model, (*ignored_dims, *idx))

    model = restore_model(model, state_dict)

    return rfield


def mask_to_boolean(mask):
    boolean = np.zeros_like(mask)
    boolean[mask > 0] = 1
    return boolean


def scale_to_one(a, axis):
    eps = 1e-16
    return a / (np.amax(a, axis=axis, keepdims=True) + eps)



def smallest_k(z, k, axis=-1):
    """
    z: D, D, C2*C2, C1
    """
    if k == 1:
        return z
    z2 = np.partition(z, k, axis=axis)
    return z2[:,:,:k]
