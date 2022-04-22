import conv_layer
import local_layer
import torch
from torch import nn
from pprintpp import pprint
import timeit


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
    run_benchmark(N, params, img_shape, trials)
