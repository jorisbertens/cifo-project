import numpy as np


def one_point_crossover(p1_r, p2_r, random_state):
    '''
    One random point is chosen to cut the individuals and the parts are shuffled to create the offsprings (Mitchell, 1999).
    '''
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

def two_point_crossover(p1_r, p2_r, random_state):
    '''
    Two random points are chosen to cut the individuals and the parts are shuffled to create the offsprings (Mitchell, 1999).
    '''
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

def parametrized_two_point_crossover(n):
    def two_point_crossover(p1_r, p2_r, random_state):
        size = min(len(p1_r), len(p2_r))
        off1_r = p1_r.copy()
        off2_r = p2_r.copy()
        for iteration in range(0,n):
            off1_r_replica = p1_r.copy()
            off2_r_replica = p2_r.copy()

            cxpoint1 = random_state.randint(1, size)
            cxpoint2 = random_state.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            off1_r = np.concatenate((off1_r_replica[0:cxpoint1], off2_r_replica[cxpoint1:cxpoint2], off1_r_replica[cxpoint2:size]))
            off2_r = np.concatenate((off2_r_replica[0:cxpoint1], off1_r_replica[cxpoint1:cxpoint2], off2_r_replica[cxpoint2:size]))
        return off1_r, off2_r
    return two_point_crossover

# TODO cycle_crossover
def cycle_crossover(p1_r, p2_r, random_state):
    return 0

# TODO ordered_crossover
def ordered_crossover(p1_r, p2_r, random_state):
    return 0

# TODO partially_matched_crossover
def partially_matched_crossover(p1_r, p2_r, random_state):
    return 0

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
# TODO is it not okay to have only one ? i think thats what its all about !!!

def geometric_crossover(p1_r, p2_r, random_state):
    random_array = random_state.uniform(size=len(p1_r))
    off1_r = ((p1_r * random_array)+ ( p2_r * (1 - random_array)))
    off2_r = ((p1_r * (1 - random_array))+ ( p2_r * random_array))
    return off1_r, off2_r

def arithmetic_threesome_crossover(p1_r, p2_r, p3_r, random_state):
    off1_r = (p1_r + p2_r)/2
    off2_r = (p1_r + p3_r)/2
    off3_r = (p2_r + p3_r)/2
    return off1_r, off2_r, off3_r

