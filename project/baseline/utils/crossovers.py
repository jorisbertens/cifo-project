import numpy as np


def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

def two_point_crossover(p1_r, p2_r, random_state):
    size = min(len(p1_r), len(p2_r))
    cxpoint1 = random_state.randint(1, size)
    cxpoint2 = random_state.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    off1_r = np.concatenate((p1_r[0:cxpoint1], p2_r[cxpoint1:cxpoint2], p1_r[cxpoint2:size]))
    off2_r = np.concatenate((p2_r[0:cxpoint1], p1_r[cxpoint1:cxpoint2], p2_r[cxpoint2:size]))
    return off1_r, off2_r

def parameterized_uniformSwap(p):
    def uniformSwap(p1_r, p2_r, random_state):
        off1_r = p1_r.copy()
        off2_r = p2_r.copy()

        size = min(len(off1_r), len(off2_r))
        for i in range(size):
            if random_state.uniform() < p:
                off1_r[i], off2_r[i] = off2_r[i], off1_r[i]

        return off1_r, off2_r
    return uniformSwap


def arithmetic_crossover(p1_r, p2_r, random_state):
    off1_r = (p1_r + p2_r)/2
    return off1_r, off1_r.copy()

