import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np



def gather_nd(params, indices, name=None):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    
    return torch.take(params, idx)

def tensordot(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)