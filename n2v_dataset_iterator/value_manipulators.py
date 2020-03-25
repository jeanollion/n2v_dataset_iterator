import numpy as np

# MOST CODE FROM NOISE TO VOID: https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py

def pm_cst(value = 0):
    def cst_fun(patch, coords, dims):
        return [value] * len(coords[0])
    return null_fun

def pm_min():
    def min_fun(patch, coords, dims):
        vmin = patch.min()
        return [vmin] * len(coords[0])
    return min_fun

def pm_normal_withoutCP():
    def normal_withoutCP(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            rand_coords = random_neighbor(patch.shape, coord)
            vals.append(patch[tuple(rand_coords)])
        return vals
    return normal_withoutCP


def pm_uniform_withCP(local_sub_patch_radius):
    def random_neighbor_withCP_uniform(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withCP_uniform


def pm_normal_additive(pixel_gauss_sigma):
    def pixel_gauss(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            vals.append(np.random.normal(patch[tuple(coord)], pixel_gauss_sigma))
        return vals
    return pixel_gauss


def pm_normal_fitted(local_sub_patch_radius):
    def local_gaussian(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord, local_sub_patch_radius)
            axis = tuple(range(dims))
            vals.append(np.random.normal(np.mean(sub_patch, axis=axis), np.std(sub_patch, axis=axis)))
        return vals
    return local_gaussian


def pm_identity(local_sub_patch_radius):
    def identity(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            vals.append(patch[coord])
        return vals
    return identity

def random_neighbor(shape, coord):
    rand_coords = sample_coords(shape, coord)
    while np.any(rand_coords == coord):
        rand_coords = sample_coords(shape, coord)

    return rand_coords


def sample_coords(shape, coord, sigma=4):
    return [normal_int(c, sigma, s) for c, s in zip(coord, shape)]


def normal_int(mean, sigma, w):
    return int(np.clip(np.round(np.random.normal(mean, sigma)), 0, w - 1))

def get_subpatch(patch, coord, local_sub_patch_radius):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1

    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift

    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[tuple(slices)]
