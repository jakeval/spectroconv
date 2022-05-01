import torch


def max_norm(layers, max_val=5):
    with torch.no_grad():
        for layer in layers:
            w = layer.weight
            norm = w.norm(2).clamp(min=max_val / 2)
            desired = torch.clamp(norm, max=max_val)
            w *= (desired / norm)
