import numpy as np

def enter(mask):
    mask = np.asarray(mask).astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    out[1:] = mask[1:] & ~mask[:-1]
    return out

def exit(mask):
    mask = np.asarray(mask).astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    out[1:] = ~mask[1:] & mask[:-1]
    return out
