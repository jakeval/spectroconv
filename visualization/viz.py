import numpy as np
from visualization import utils


def combining_weights_activation_map(x, model, lrlc_layer, cw_layer):
    """
    X: N, C2, H, W

    For LRLC layers, view the activation map of each of the R basis kernels.
    
    Return shape: N x R x C2 x H x W
    """
    N, _, H, W = x.shape
    rfield = utils.layer_receptive_field((H,W), 2, model, lrlc_layer)
    Lh, Lw = rfield.shape[0], rfield.shape[1]
    activation = utils.LayerActivationMonitor(cw_layer)
    _ = model(x)
    layer_output = activation.get_layer_output().detach().numpy()
    N_cw, R, Lh_cw, Lw_cw = layer_output.shape # Some of the dimensions may be truncated to 1.
    layer_output = np.tile(layer_output, (N // N_cw, 1, Lh // Lh_cw, Lw // Lw_cw))

    mask = np.zeros((N, R, H, W))

    for n in range(N):
        for r in range(R):
            for lh in range(Lh):
                for lw in range(Lw):
                    new_mask = layer_output[n,r,lh,lw] * rfield[lh,lw]
                    mask[n,r] = np.maximum(mask[n,r], new_mask)
    
    mask = utils.scale_to_one(mask, (2,3))
    return mask, layer_output


def basis_activation_map(x, model, layer):
    """
    X: N, C2, H, W

    For LRLC layers, view the activation map of each of the R basis kernels.
    
    Return shape: N x R x C2 x H x W
    """
    N, _, H, W = x.shape
    rfield = utils.layer_receptive_field((H,W), 2, model, layer)
    activation = utils.BasisActivationMonitor(layer)
    _ = model(x)
    layer_output = activation.get_layer_output().detach().numpy() # N, R, C2, Lh, Lw
    _, R, C2, Lh, Lw = layer_output.shape

    mask = np.zeros((N, R, C2, H, W))

    for n in range(N):
        for r in range(R):
            for c2 in range(C2):
                for lh in range(Lh):
                    for lw in range(Lw):
                        new_mask = layer_output[n,r,c2,lh,lw] * rfield[lh,lw]
                        mask[n,r,c2] = np.maximum(mask[n,r,c2], new_mask)

    mask = utils.scale_to_one(mask, (3,4))
    return mask, layer_output


def convolution_activation_map(x, model, layer):
    """
    X: N, C2, H, W

    For any layer which outputs an N x C2 x Lh x Lw feature map,
    for each pixel in the feature map output, overlay its activation level
    onto the input image by masking with its receptive field.

    Return shape: N x C2 x H x W
    """
    N, _, H, W = x.shape
    rfield = utils.layer_receptive_field((H,W), 2, model, layer)
    activation = utils.LayerActivationMonitor(layer)
    _ = model(x)
    layer_output = activation.get_layer_output().detach().numpy() # N, C2, Lh, Lw
    _, C2, Lh, Lw = layer_output.shape

    mask = np.zeros((N, C2, H, W))

    for n in range(N):
        for c2 in range(C2):
            for lh in range(Lh):
                for lw in range(Lw):
                    new_mask = layer_output[n,c2,lh,lw] * rfield[lh,lw]
                    mask[n,c2] = np.maximum(mask[n,c2], new_mask)
    
    mask = utils.scale_to_one(mask, (2,3))
    return mask, layer_output


def feature_distance(w):
    """
    w: D, C2, C1, K

    1. Find the pairwise distances between all output dimensions (D) and filters (C2).
       > this is done by treating D dimensions and C2 filters as one dimension
       > the input kernel block (K) and input filters (C1) are aligned
    2. For any given pair of dimensions, try to match the filters by smallest distance
       > we cheat: instead of computing a 1-to-1 mapping of C2 -> C2 filters, just take the 
         C2 pairs with smallest pairwise distance. It's an approximation
       > this is done on a per-input filter basis. i.e. we get D*D*C1 different mappings
    3. For every mapping of C2->C2 filters, compute its average distance
    4. For every dimension pair, compute the average mapping distance
       > Each pair of dimensions has one mapping per input filter.
    """
    D, C2, C1, K = w.shape
    w2 = w.reshape(D*C2, C1, K)

    # step 1
    dist = ((w2[:,None,:] - w2)**2).sum(axis=-1) # D*C2, D*C2, C1
    dist = dist.reshape(D,C2,D,C2,C1).swapaxes(1,2) # D, D, C2, C2, C1
    # step 2
    z = utils.smallest_k(dist.reshape(D, D, C2 * C2, C1), C2, axis=2) # D, D, C2, C1
    # step 3
    z = z.mean(axis=-1) # D, D, C2
    # step 4
    z = z.mean(axis=-1) # D, D
    return z


def _feature_distance_broken(w):
    """
    w: D, C2, C1, K

    I'm leaving this function as a reminder that it doesn't work.

    It is much more efficient, but is overly lenient.
    It fails in the mapping approximation described in step (2) of the unoptimized version.
    Instead of computing D*D*C1 mappings, we only compute D*D mappings.
    So for a d,d pair, instead of averaging over all C1 mappings, we are constructing a single mapping
        of C2 -> C2 filters from all possible C2*C2*C1 pairs. This is cheating! We can pick-and-choose
        the mapping from a greater number of options.

    If you can solve for an exact 1-1 mapping, this will work. The problem:
        > given an N,N matrix, find N numbers such that their sum is minimized and
          no two share a row/column
        > I think this is NP-Complete
    """
    D, C2, C1, K = w.shape
    w2 = w.reshape(D*C2, C1*K)

    # step 1
    #dist = metrics.pairwise.euclidean_distances(w2) # D*C2, D*C2
    dist = dist.reshape(D,C2,D,C2).swapaxes(1,2) # D, D, C2, C2
    # step 2
    z = utils.smallest_k(dist.reshape(D, D, C2 * C2), C2) # D, D, C2
    # step 3
    return z.mean(axis=-1) # D, D, C2
