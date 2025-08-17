import numpy as np

def catmull_rom_one_segment(p0, p1, p2, p3, n_points):
    t = np.linspace(0, 1, n_points).reshape(-1, 1)   # (n,1) per broadcasting
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )

def catmull_rom_chain(P, n_points):
    P = np.array(P)
    if len(P) < 4:
        return P
    segs = []
    for i in range(len(P) - 3):
        segs.append(catmull_rom_one_segment(P[i], P[i+1], P[i+2], P[i+3], n_points))
    return np.vstack(segs)
