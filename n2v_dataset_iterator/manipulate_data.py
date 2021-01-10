# CODE ADAPTED FROM NOISE TO VOID: https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py
import numpy as np
from .value_manipulators import pm_uniform_withCP

def manipulate_data_fun(patch=True, shape=(32, 32), perc_pix=3.125, value_manipulation=pm_uniform_withCP(5), weighted_loss = True, full_output=False):
    def fun(batch):
        shape_ = shape
        if shape_ is None or not patch:
            shape_ = batch.shape[1:-1]
        if patch:
            X_out = np.zeros( (batch.shape[0], *shape_, batch.shape[-1]), dtype=batch.dtype)
        else:
            X_out = None
        Y_out = np.zeros( (batch.shape[0], *shape_, batch.shape[-1]*2), dtype=batch.dtype)
        manipulate_data(batch, X_out, Y_out, shape_, perc_pix, value_manipulation, weighted_loss, full_output)
        if patch:
            return X_out, Y_out
        else:
            return Y_out
    return fun

def manipulate_data(X, X_out, Y_out, shape, perc_pix=3.125, value_manipulation=pm_uniform_withCP(5),  weighted_loss = True,full_output=False):
    dims = len(shape)
    sampling_range = np.array(X.shape[1:-1]) - np.array(shape)
    num_pix = np.product(shape)/100.0 * perc_pix
    assert num_pix>=1, "at least one pixel should be manipulated"
    box_size = np.ceil(np.sqrt(100.0/perc_pix)).astype(np.int)
    if dims == 2:
        patch_sampler = subpatch_sampling2D
        get_stratified_coords = get_stratified_coords2D
    elif dims == 3:
        patch_sampler = subpatch_sampling3D
        get_stratified_coords = get_stratified_coords3D
    else:
        raise ValueError("dimension number not supported")
    n_chan = X.shape[-1]
    offsets = get_offset(box_size, shape)
    if X_out is not None:
        patch_sampler(X, X_out, sampling_range, shape)
    else:
        X_out = X
    #if not full_output:
    #    Y_out *= 0
    if full_output:
        np.copyto(Y_out[...,0:n_chan], X_out)
    n_pix = float(np.prod(X.shape[1:-1]))
    for j in range(X.shape[0]):
        coords = get_stratified_coords(box_size, offsets, shape)
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            if not full_output:
                Y_out[indexing] = X_out[indexing]
            x = value_manipulation(X_out[j, ..., c], coords, dims)
            if weighted_loss:
                Y_out[indexing_mask] = n_pix / len(coords[0]) # modification from original code so that only loss = mask * MAE/MSE-loss
            else:
                Y_out[indexing_mask] = 1
            X_out[indexing] = x

def subpatch_sampling2D(X, X_Batches, sampling_range, shape):
    if sampling_range[0]==0 and sampling_range[1]==0:
        for i in range(X.shape[0]):
            X_Batches[i] = np.copy(X[i])
    else:
        y_start = np.random.randint(0, sampling_range[0] + 1, size=X.shape[0])
        x_start = np.random.randint(0, sampling_range[1] + 1, size=X.shape[0])
        for i in range(X.shape[0]):
            X_Batches[i] = np.copy(X[i, y_start[i]:y_start[i] + shape[0], x_start[i]:x_start[i] + shape[1]])

def subpatch_sampling3D(X, X_Batches, sampling_range, shape):
    if sampling_range[0]==0 and sampling_range[1]==0 and sampling_range[2]==0:
        for i in range(X.shape[0]):
            X_Batches[i] = np.copy(X[i])
    else:
        z_start = np.random.randint(0, sampling_range[0] + 1, size=X.shape[0])
        y_start = np.random.randint(0, sampling_range[1] + 1, size=X.shape[0])
        x_start = np.random.randint(0, sampling_range[2] + 1, size=X.shape[0])
        for i in range(X.shape[0]):
            X_Batches[i] = np.copy(X[i, z_start[i]:z_start[i] + shape[0], y_start[i]:y_start[i] + shape[1], x_start[i]:x_start[i] + shape[2]])

def get_offset(box_size, shape):
    offsets = [np.arange(int(np.ceil(s / box_size))) * box_size for s in shape]
    return [a.flatten() for a in np.meshgrid(*offsets, sparse=False, indexing='ij')]

def get_stratified_coords2D(box_size, offset_yx, shape):
    x = np.random.uniform(size=offset_yx[0].shape[0]) * box_size
    y = np.random.uniform(size=offset_yx[0].shape[0]) * box_size
    y = y.astype(np.int) + offset_yx[0]
    x = x.astype(np.int) + offset_yx[1]
    # remove coords outside image
    mask = np.logical_and(y<shape[0], x<shape[1])
    return (y[mask], x[mask])

def get_stratified_coords3D(box_size, offset_zyx, shape): # TODO test
    x = np.random.uniform(size=offset_zyx[0].shape[0]) * box_size
    y = np.random.uniform(size=offset_zyx[0].shape[0]) * box_size
    z = np.random.uniform(size=offset_zyx[0].shape[0]) * box_size
    z = z.astype(np.int) + offset_zyx[0]
    y = y.astype(np.int) + offset_zyx[1]
    x = x.astype(np.int) + offset_zyx[2]
    # remove coords outside image
    mask = (z<shape[0]) & (y<shape[1]) & (x<shape[2])
    return (z[mask], y[mask], x[mask])
