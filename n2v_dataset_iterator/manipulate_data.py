import numpy as np

# CODE ADAPTED FROM NOISE TO VOID: https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py

def manipulate_data_fun(perc_pix=0.198, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    def fun(batch):
        Y = np.zeros(batch.shape[:-1]+(batch.shape[-1]*2), dtype=batch.dtype)
        return manipulate_data(X, Y, perc_pix, shape, value_manipulation)
    return fun

def manipulate_data(X, Y, perc_pix=0.198, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords2D
        rand_float = rand_float_coords2D(box_size)
    elif dims == 3:
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = get_stratified_coords3D
        rand_float = rand_float_coords3D(box_size)

    n_chan = X.shape[-1]

    Y *= 0
    for j in range(X.shape[0]):
        coords = get_stratified_coords(rand_float, box_size=box_size, shape=np.array(X.shape)[1:-1])
        for c in range(n_chan):
            indexing = (j,) + coords + (c,)
            indexing_mask = (j,) + coords + (c + n_chan,)
            y = X[indexing]
            x = value_manipulation(X[j, ..., c], coords, dims)
            Y[indexing] = y
            Y[indexing_mask] = 1. / coords.shape[0] # modification from original code so that only loss = mask * loss
            X[indexing] = x

def get_subpatch(patch, coord, local_sub_patch_radius):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift

    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[tuple(slices)]

def subpatch_sampling2D(X, X_Batches, indices, range, shape):
    for j in indices:
        y_start = np.random.randint(0, range[0] + 1)
        x_start = np.random.randint(0, range[1] + 1)
        X_Batches[j] = np.copy(X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]])

def subpatch_sampling3D(X, X_Batches, indices, range, shape):
    for j in indices:
        z_start = np.random.randint(0, range[0] + 1)
        y_start = np.random.randint(0, range[1] + 1)
        x_start = np.random.randint(0, range[2] + 1)
        X_Batches[j] = np.copy(X[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]])

def get_offset(shape, boxsize): # TODO TEST
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    offset_y = np.zeros(shape = (box_count_y, box_count_x), dtype=np.int)
    offset_x = np.zeros(shape = (box_count_y, box_count_x), dtype=np.int)

    for y in range(1, box_count_y):
        offset_y[y, :] = y * boxsize
    for x in range(1, box_count_x):
        offset_x[:, x] = x * boxsize
    return offset_y.flatten(), offset_x.flatten()

def get_stratified_coords2D_2(boxsize, offset_y, offset_x, shape): # TODO TEST
    x = np.random.uniform(size=offset_y.shape[0]) * boxsize
    y = np.random.uniform(size=offset_y.shape[0]) * boxsize
    x = x.astype(np.int) + offset_x
    y = y.astype(np.int) + offset_y
    # remove coords outside image
    mask = (y<shape[0]) & x<shape[1])
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
