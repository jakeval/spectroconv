import torch
from layers import lrlc_layer


def max_norm(layers, max_val=5):
    with torch.no_grad():
        for layer in layers:
            w = None
            if isinstance(layer, lrlc_layer.LowRankLocallyConnected):
                w = layer.weight_bases
            else:
                w = layer.weight
            norm = w.norm(2).clamp(min=max_val / 2)
            desired = torch.clamp(norm, max=max_val)
            w *= (desired / norm)
