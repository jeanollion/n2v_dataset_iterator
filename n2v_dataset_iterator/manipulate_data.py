# CODE ADAPTED FROM NOISE TO VOID: https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py
import numpy as np
from .value_manipulators import pm_uniform_withCP

def manipulate_data_fun(patch=True, shape=(32, 32), perc_pix=0.198, value_manipulation=pm_uniform_withCP(5), full_output=False):
    def fun(batch):
        shape_ = shape
        if shape_ is None or not patch:
            shape_ = batch.shape[1:-1]
        if patch:
            X_out = np.zeros( (batch.shape[0], *shape_, batch.shape[-1]), dtype=batch.dtype)
        else:
            X_out = None
        Y_out = np.zeros( (batch.shape[0], *shape_, batch.shape[-1]*2), dtype=batch.dtype)
        manipulate_data(batch, X_out, Y_out, shape_, perc_pix, value_manipulation, full_output)
        if patch:
            return X_out, Y_out
        else:
            return Y_out
    return fun

def manipulate_data(X, X_out, Y_out, shape=(32, 64), perc_pix=0.198, value_manipulation=pm_uniform_withCP(5), full_output=False):
    dims = len(shape)
    sampling_range = np.array(X.shape[1:-1]) - np.array(shape)
    if dims == 2:
        patch_sampler = subpatch_sampling2D
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords2D_2
        rand_float = rand_float_coords2D(box_size)
    elif dims == 3:
        patch_sampler = subpatch_sampling3D
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords3D
        rand_float = rand_float_coords3D(box_size)

    n_chan = X.shape[-1]
    offset_y, offset_x = get_offset(box_size, shape)
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
        #coords = get_stratified_coords(rand_float, box_size=box_size, shape=np.array(X.shape)[1:-1])
        coords = get_stratified_coords(box_size, offset_y, offset_x, shape)
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            if not full_output:
                Y_out[indexing] = X_out[indexing]
            x = value_manipulation(X_out[j, ..., c], coords, dims)
            Y_out[indexing_mask] = n_pix / len(coords[0]) # modification from original code so that only loss = mask * MAE/MSE-loss
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

def get_offset(box_size, shape): # TODO TEST
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    offset_y = np.zeros(shape = (box_count_y, box_count_x), dtype=np.int)
    offset_x = np.zeros(shape = (box_count_y, box_count_x), dtype=np.int)

    for y in range(1, box_count_y):
        offset_y[y, :] = y * box_size
    for x in range(1, box_count_x):
        offset_x[:, x] = x * box_size
    return offset_y.flatten(), offset_x.flatten()

def get_stratified_coords2D_2(box_size, offset_y, offset_x, shape): # TODO TEST
    x = np.random.uniform(size=offset_y.shape[0]) * box_size
    y = np.random.uniform(size=offset_y.shape[0]) * box_size
    x = x.astype(np.int) + offset_x
    y = y.astype(np.int) + offset_y
    # remove coords outside image
    mask = (y<shape[0]) & (x<shape[1])
    return (y[mask], x[mask])

def get_stratified_coords2D(coord_gen, box_size, shape):
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = next(coord_gen)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)

def get_stratified_coords3D(coord_gen, box_size, shape):
    box_count_z = int(np.ceil(shape[0] / box_size))
    box_count_y = int(np.ceil(shape[1] / box_size))
    box_count_x = int(np.ceil(shape[2] / box_size))
    x_coords = []
    y_coords = []
    z_coords = []
    for i in range(box_count_z):
        for j in range(box_count_y):
            for k in range(box_count_x):
                z, y, x = next(coord_gen)
                z = int(i * box_size + z)
                y = int(j * box_size + y)
                x = int(k * box_size + x)
                if (z < shape[0] and y < shape[1] and x < shape[2]):
                    z_coords.append(z)
                    y_coords.append(y)
                    x_coords.append(x)
    return (z_coords, y_coords, x_coords)


def rand_float_coords2D(boxsize):
    while True:
        yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

def rand_float_coords3D(boxsize):
    while True:
        yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)
