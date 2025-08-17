import numpy as np

def compress_memory(mem, max_len):
    if len(mem) <= max_len:
        return mem
    # togli il punto più vecchio o uno scelto con log-spacing
    remove_idx = int(np.logspace(0, np.log10(len(mem)), max_len, endpoint=False)[0])
    mem.pop(remove_idx)
    return mem
