import sys
sys.path.insert(1, '../')

from layers import conv_layer
from layers import local_layer
from layers import lrlc_layer
import torch
from torch import nn
from torch.nn import functional as F
from pprintpp import pprint
import timeit
import numpy as np


def rel_err(ytrue, ytest):
    return torch.abs((ytest - ytrue) / ytrue).mean().item()


def run_benchmark(N, params, img_shape, trials):
    print("-"*30, "Check runtime benchmarks", "-"*30)
    H, W = img_shape
    X = torch.rand((N, params['C1'], H, W), dtype=torch.float32)
    torch_conv = nn.Conv2d(params['C1'], params['C2'], params['k'], stride=params['s'], padding=params['p']).float()
    custom_conv = conv_layer.CustomConv(params['C1'], params['C2'], params['k'], stride=params['s'], padding=params['p']).float()
    custom_local = local_layer.LocallyConnected(X.shape, params['C1'], params['C2'], params['k'], stride=params['s'], padding=params['p'])
    timing_dict = {
        'torch_conv': torch_conv,
        'custom_conv': custom_conv,
        'local_conv': custom_local
    }
    for key, layer in timing_dict.items():
        timing_dict[key] = benchmark(layer, X, trials)
    pprint(timing_dict)
    return timing_dict


def benchmark(layer, X, trials):
    def run_test(conv, X):
        Z = conv(X)
        l = Z.mean()
        l.backward()

    time = timeit.timeit(lambda: run_test(layer, X), number=trials)
    return time / trials


def check_conv():
    print("-"*30, "Check custom convolution accuracy", "-"*30)
    params_to_test = [
        {
            'in_channels': 1,
            'out_channels': 1,
            'kernel_size': 3,
            'stride': 1,
            'padding': 0
        },
        {
            'in_channels': 1,
            'out_channels': 1,
            'kernel_size': (3,5),
            'stride': 1,
            'padding': 0
        },
        {
            'in_channels': 1,
            'out_channels': 1,
            'kernel_size': 3,
            'stride': (2,3),
            'padding': 0
        },
        {
            'in_channels': 1,
            'out_channels': 1,
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same'
        },
        {
            'in_channels': 5,
            'out_channels': 8,
            'kernel_size': 3,
            'stride': 1,
            'padding': 'same'
        },
    ]

    N = 5
    H = 30
    W = 50
    for params in params_to_test:
        true_conv = nn.Conv2d(**params).float()
        test_conv = conv_layer.CustomConv(**params).float()
        with torch.no_grad():
            true_conv.weight.copy_(test_conv.weight)
            true_conv.bias.copy_(test_conv.bias)
        X = torch.rand((N, params['in_channels'], H, W), dtype=torch.float32)
        Z_true = true_conv(X)
        Z_test = test_conv(X)
        print("Params are:")
        pprint(params)
        print("Output err is: ", rel_err(Z_true, Z_test))
        
        l_true = Z_true.mean()
        l_true.backward()
        l_test = Z_test.mean()
        l_test.backward()

        print("Grad errors are: ")
        for p_true, p_test in zip(true_conv.parameters(), test_conv.parameters()):
            print(rel_err(p_true.grad, p_test.grad))


def check_local():
    print("-"*30, "Check local layer accuracy", "-"*30)
    N = 2
    C1 = 3
    C2 = 4
    s = 1
    p = 0
    bias=True

    X = torch.rand((N, C1, 6, 6)).double()
    lc = local_layer.LocallyConnected((6,6), C1, C2, 3, s, p, bias=bias)
    Z_test = lc(X)

    w, b = None, None
    w = lc.weight.detach().numpy()
    if bias:
        b = lc.bias.detach().numpy()
    Z_true, w, b = local_layer.naive_lc(X, C2, w, b, bias=bias)

    print(rel_err(Z_true, Z_test.detach()))


def check_hybrid_freq():
    print("-"*30, "Check hybrid-F accuracy", "-"*30)
    N = 2
    C1 = 3
    C2 = 4
    s = 1
    p = 0
    bias=True
    local_dim = 0

    X = torch.rand((N, C1, 6, 6)).double()
    hc = local_layer.LocallyConnectedConv(local_dim, (6,6), C1, C2, 3, s, p, bias=bias)
    Z_test = hc(X)

    w, b = None, None
    w = hc.weight.detach().numpy()
    if bias:
        b = hc.bias.detach().numpy()
    Z_true, w, b = local_layer.naive_hc(local_dim, X.numpy(), C2, w, b, bias=bias)

    print(rel_err(Z_true, Z_test.detach()))


def check_hybrid_time():
    print("-"*30, "Check hybrid-T accuracy", "-"*30)
    N = 2
    C1 = 3
    C2 = 4
    s = 1
    p = 0
    bias=True
    local_dim = 1

    X = torch.rand((N, C1, 6, 6)).double()
    hc = local_layer.LocallyConnectedConv(local_dim, (6,6), C1, C2, 3, s, p, bias=bias)
    Z_test = hc(X)

    w, b = None, None
    w = hc.weight.detach().numpy()
    if bias:
        b = hc.bias.detach().numpy()
    Z_true, w, b = local_layer.naive_hc(local_dim, X.numpy(), C2, w, b, bias=bias)

    print(rel_err(Z_true, Z_test.detach()))


def check_lrlc():
    print("-"*30, "Check lrlc accuracy", "-"*30)
    N = 2
    C1 = 2
    C2 = 2
    s = 1
    p = 0
    bias=True
    R = 1

    X = torch.rand((N, C1, 6, 6)).float()
    channel_bias = torch.tensor([1, 2]).reshape((2, 1, 1))
    lrlc = lrlc_layer.LowRankLocallyConnected(R, (6,6), C1, C2, 3, s, p, bias=bias).float()
    ocw = lrlc_layer.OuterCombiningWeights(R, lrlc.L)
    with torch.no_grad():
        lrlc.bias_c += channel_bias
    cw = ocw()
    weights = lrlc.get_weights(cw, lrlc.weight_bases, tile=True)[0]

    err = 0
    for h in range(4):
        for w in range(4):
            err += rel_err(lrlc.weight_bases, weights[h,w])
    print(f"Rank 1 weights err: {err/16}")

    conv = nn.Conv2d(C1, C2, kernel_size=3, stride=s, padding=p, bias=bias).float()
    conv.weight = lrlc.weight_bases
    with torch.no_grad():
        conv.bias.copy_(channel_bias.squeeze()).float()
    test_out = lrlc(X, cw)
    true_out = conv(X)
    print("Rank 1 conv err:", rel_err(true_out, test_out))

    R = 3
    lrlc = lrlc_layer.LowRankLocallyConnected(R, (6,6), C1, C2, 3, s, p, bias=bias).float()
    ocw = lrlc_layer.OuterCombiningWeights(R, lrlc.L)
    cw = ocw()
    weights = lrlc.get_weights(cw, lrlc.weight_bases, tile=True)[0]
    basis = lrlc.weight_bases.view((R, C2, C1, 3, 3))
    true_filter = basis.mean(0)
    err += rel_err(true_filter, weights[0,0])
    err += rel_err(true_filter, weights[1,1])
    print("Rank 3 uniform weights err: ", err/2)

    conv = nn.Conv2d(C1, C2, kernel_size=3, stride=s, padding=p, bias=bias).float()
    with torch.no_grad():
        lrlc.bias_c += channel_bias
        conv.weight.copy_(basis.mean(0))
        conv.bias.copy_(channel_bias.squeeze()).float()
    test_out = lrlc(X, cw)
    true_out = conv(X)
    print("Rank 3 uniform conv err:", rel_err(true_out, test_out))


    lc = local_layer.LocallyConnected((6,6), C1, C2, (3,3), s, p, bias=bias)
    Lh, Lw, _, _, Kh, Kw = weights.shape
    lc_weights = torch.moveaxis(weights, 2, 0).view(C2, Lh*Lw, C1*Kh*Kw) # Lh, Lw, C2, C1, Kh, Kw
    lc_bias = lrlc.get_bias(tile=True).view(C2, Lh*Lw)
    with torch.no_grad():
        lc.weight.copy_(lc_weights)
        lc.bias.copy_(lc_bias)
    test_out = lrlc(X, cw)
    true_out = lc(X)
    print("Rank 3 uniform LC err:", rel_err(true_out, test_out))

    with torch.no_grad():
        ocw.wts_h.copy_(torch.rand(R, Lh, 1).float())
        ocw.wts_w.copy_(torch.rand(R, 1, Lw).float())
        lrlc.bias_h.copy_(torch.rand(1, Lh, 1))
        lrlc.bias_w.copy_(torch.rand(1, 1, Lw))
        lrlc.bias_c.copy_(torch.rand(C2, 1, 1))
    cw = ocw()
    weights = lrlc.get_weights(cw, lrlc.weight_bases, tile=True)[0]
    Lh, Lw, _, _, Kh, Kw = weights.shape
    lc_weights = torch.moveaxis(weights, 2, 0).view(C2, Lh*Lw, C1*Kh*Kw) # Lh, Lw, C2, C1, Kh, Kw
    lc_bias = lrlc.get_bias(tile=True).view(C2, Lh*Lw)
    with torch.no_grad():
        lc.weight.copy_(lc_weights)
        lc.bias.copy_(lc_bias)
    test_out = lrlc(X, cw)
    true_out = lc(X)
    print("Rank 3 non-uniform LC err:", rel_err(true_out, test_out))


def check_combining_weights():
    print("-"*30, "Check localization combining weights", "-"*30)
    N = 2
    C1 = 16
    Cd = 4
    C2 = 8
    D1 = 16
    D2 = 16
    R = 3

    X = torch.rand((N, C1, D1, D2)).float()

    lrlc = lrlc_layer.LowRankLocallyConnected(R, (D1,D2), C1, C2, 3, 1, 0, bias=True).float()
    size = lrlc.L
    print("Number of lrlc parameters:", sum([np.prod(p.size()) for p in lrlc.parameters()]))

    print("Number of combining weights:", size[0]*size[1]*R)

    ocw = lrlc_layer.OuterCombiningWeights(R, lrlc.L)
    print("Number of outer product parameters:", sum([np.prod(p.size()) for p in ocw.parameters()]))

    lcw = lrlc_layer.LocalizationCombiningWeights(R, (D1, D2), size, C1, Cd)
    print("Number of localization parameters:", sum([np.prod(p.size()) for p in lcw.parameters()]))

    lccw = lrlc_layer.LocallyConnectedCombiningWeights(R, size, C1, Cd, compression=3).float()
    print("Number of locally connected parameters:", sum([np.prod(p.size()) for p in lccw.parameters()]))

    cw = lcw(X)
    z = lrlc(X, cw)

    cw = lccw(X)
    z = lrlc(X, cw)


def check_lrlc_hybrid(local_dim):
    dim_name = "F" if local_dim == 0 else "T"
    print("-"*30, f"Check lrlc-{dim_name}", "-"*30)
    N = 2
    C1 = 3
    C2 = 6
    R = 3
    Cd = 2

    X = torch.rand((N, C1, 6, 6)).float()
    lrlc = lrlc_layer.LowRankLocallyConnected(R, (6,6), C1, C2, local_dim=local_dim).float()
    lc = local_layer.LocallyConnected((6,6), C1, C2).float()
    Lh, Lw = lrlc.L

    print("OUTER CW")
    ocw = lrlc_layer.OuterCombiningWeights(R, lrlc.L, local_dim=local_dim).float()
    with torch.no_grad():
        if local_dim is None or local_dim == 0:
            ocw.wts_h.copy_(torch.rand(R, Lh, 1).float())
            lrlc.bias_h.copy_(torch.rand(1, Lh, 1))
        if local_dim is None or local_dim == 1:
            ocw.wts_w.copy_(torch.rand(R, 1, Lw).float())
            lrlc.bias_c.copy_(torch.rand(C2, 1, 1))
        lrlc.bias_c.copy_(torch.rand(C2, 1, 1))
    cw = ocw()
    weights = lrlc.get_weights(cw, lrlc.weight_bases, tile=True)[0]
    _, _, _, _, Kh, Kw = weights.shape
    lc_weights = torch.moveaxis(weights, 2, 0).view(C2, Lh*Lw, C1*Kh*Kw) # Lh, Lw, C2, C1, Kh, Kw
    lc_bias = lrlc.get_bias(tile=True).view(C2, Lh*Lw)
    with torch.no_grad():
        lc.weight.copy_(lc_weights)
        lc.bias.copy_(lc_bias)
    test_out = lrlc(X, cw)
    true_out = lc(X)
    print("OCW LC err:", rel_err(true_out, test_out))
    if local_dim is not None:
        shared_dim = int(not bool(local_dim))
        print(f"Shared deviation (should be zero):", weights.std(axis=shared_dim).mean().item())
        print(f"Local deviation (should be high):", weights.std(axis=local_dim).mean().item())

    print("\nLOCALIZATION CW")
    lcw = lrlc_layer.LocalizationCombiningWeights(R, (6,6), lrlc.L, C1, Cd, local_dim=local_dim).float()
    with torch.no_grad():
        if local_dim is None or local_dim == 0:
            lrlc.bias_h.copy_(torch.rand(1, Lh, 1))
        if local_dim is None or local_dim == 1:
            lrlc.bias_c.copy_(torch.rand(C2, 1, 1))
        lrlc.bias_c.copy_(torch.rand(C2, 1, 1))
    cw = lcw(X)
    cw_noise = torch.randn_like(cw)
    cw = cw + cw_noise # add noise because by values are mostly uniform on initialization
    cw = F.softmax(cw, dim=1)

    weights = lrlc.get_weights(cw, lrlc.weight_bases, tile=True)
    test_out = lrlc(X, cw)
    for n in range(N):
        w = weights[n]
        _, _, _, _, Kh, Kw = w.shape
        lc_weights = torch.moveaxis(w, 2, 0).view(C2, Lh*Lw, C1*Kh*Kw) # Lh, Lw, C2, C1, Kh, Kw
        lc_bias = lrlc.get_bias(tile=True).view(C2, Lh*Lw)
        with torch.no_grad():
            lc.weight.copy_(lc_weights)
            lc.bias.copy_(lc_bias)
        true_out = lc(X)
        print("LCW LC err:", rel_err(true_out[n], test_out[n]))
        if local_dim is not None:
            shared_dim = int(not bool(local_dim))
            print(f"Shared deviation (should be zero):", w.std(axis=shared_dim).mean().item())
            print(f"Local deviation (should be high):", w.std(axis=local_dim).mean().item())


if __name__ == '__main__':
    #check_conv()
    params = {
        'C1': 3,
        'C2': 8,
        'k': 3,
        's': 1,
        'p': 'same'
    }
    check_local()
    check_hybrid_freq()
    check_hybrid_time()
    N = 32
    img_shape = (128, 71)
    trials = 5
    #run_benchmark(N, params, img_shape, trials)
    check_combining_weights()
    check_lrlc()
    check_lrlc_hybrid(0)
    check_lrlc_hybrid(1)
