from itertools import product

def build_index(ndim, *groups):
    idx = [0] * ndim

    for axes, values in groups:
        for axis, value in zip(axes, values):
            idx[axis] = value

    return tuple(idx)

def indices(shape): yield from product(*(range(s) for s in shape)) if shape != () else ()

def zeroes(shape):
    if len(shape) == 0:
        return 0

    return [zeroes(shape[1:]) for _ in range(shape[0])]

def set_nested(lst, idx, value):
    for i in idx[:-1]:
        lst = lst[i]
    lst[idx[-1]] = value
