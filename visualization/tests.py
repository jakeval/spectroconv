import sys
sys.path.insert(1, '../')

from models import lrlc_model_debug
from visualization import viz, utils
import numpy as np
import matplotlib.pyplot as plt
import torch


def shuffle_dim(x, dim):
    """Taken from https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    """
    idx = np.random.rand(*x.shape).argsort(axis=dim)
    return np.take_along_axis(x, idx, axis=dim)


def test_distance():
    print("-"*30, "Test Distance", "-"*30)
    D = 32
    C2 = 8
    C1 = 3
    K = 9
    x1 = np.random.random((C2, C1, K))
    x2 = np.random.random((C2, C1, K)) * 10
    x = np.array([a*x1 + (1-a)*x2 for a in np.linspace(0,1,D)])

    x_c2 = shuffle_dim(x, 1)
    S = viz.feature_distance(x_c2)
    plt.imshow(-S)
    plt.show()


def test_receptive_field():
    print("-"*30, "Test Receptive Field", "-"*30)
    D1, D2 = 64, 64
    model = lrlc_model_debug.LrlcClfDebug(5, (D1, D2), [None]*3).float()
    layer = model._conv_layers[-1]
    rfield = utils.layer_receptive_field((D1, D2), 2, model, layer)
    plt.imshow(rfield[0,0])
    plt.show()
    plt.imshow(rfield[4,4])
    plt.show()


def test_cnn_activation_map():
    print("-"*30, "Test CNN Activations", "-"*30)
    D1, D2 = 64, 64
    N = 5
    x = torch.rand((N, 1, D1, D2))
    model = lrlc_model_debug.LrlcClfDebug(5, (D1, D2), [None]*3).float()
    layer = model.lrlc_layer
    map, _ = viz.convolution_activation_map(x, model, layer) # N, C2, H, W
    plt.imshow(map[0,0])
    plt.show()


def test_basis_activation_map():
    print("-"*30, "Test Basis Activations", "-"*30)
    D1, D2 = 64, 64
    N = 5
    x = torch.rand((N, 1, D1, D2))
    model = lrlc_model_debug.LrlcClfDebug(5, (D1, D2), [None]*3).float()
    layer = model.lrlc_layer
    map, _ = viz.basis_activation_map(x, model, layer) # N, R, C2, H, W
    plt.imshow(map[0,0,0])
    plt.show()


def test_combining_weights_activation_map():
    print("-"*30, "Test Combining Weights Activations", "-"*30)
    D1, D2 = 64, 64
    R = 4
    N = 5
    x = torch.rand((N, 1, D1, D2))
    model = lrlc_model_debug.LrlcClfDebug(R, (D1, D2), [None]*3, local_dim=None).float()
    lrlc_layer = model.lrlc_layer
    cw_layer = model.combining_weights
    cw_layer._random_init()
    map, _ = viz.combining_weights_activation_map(x, model, lrlc_layer, cw_layer) # N, R, H, W

    plt.imshow(map[0,0])
    plt.show()


def test_combining_weights_activation_map_lcw():
    print("-"*30, "Test Combining Weights Activations", "-"*30)
    D1, D2 = 64, 64
    R = 4
    N = 5
    x = torch.rand((N, 1, D1, D2))
    model = lrlc_model_debug.LrlcClfDebug(R, (D1, D2), [None]*3, local_dim=None, lcw=True).float()
    test = model(x)
    lrlc_layer = model.lrlc_layer
    cw_layer = model.combining_weights
    cw_layer._random_init()
    map, out = viz.combining_weights_activation_map(x, model, lrlc_layer, cw_layer) # N, R, H, W

    plt.imshow(out[0,0])
    plt.show()

    plt.imshow(out[1,0])
    plt.show()

    plt.imshow(map[0,0])
    plt.show()


if __name__ == '__main__':
    test_distance()
    test_receptive_field()
    test_cnn_activation_map()
    test_basis_activation_map()
    test_combining_weights_activation_map()
    test_combining_weights_activation_map_lcw()
