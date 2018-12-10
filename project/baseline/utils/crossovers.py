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
    size = min(len(p1_r), len(p2_r))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[p1_r[i]] = i
        p2[p2_r[i]] = i
    # Choose crossover points
    cxpoint1 = random_state.randint(0, size)
    cxpoint2 = random_state.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in xrange(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

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


def parameterized_random_crossover(swap_p):
    def random_crossover(p1_r, p2_r, random_state):
        crossovers = [
            arithmetic_crossover,
            parameterized_uniformSwap(swap_p),
            one_point_crossover,
            two_point_crossover
        ]
        crossover_algo = random_state.choice(crossovers)
        return crossover_algo(p1_r, p2_r, random_state)
    return random_crossover