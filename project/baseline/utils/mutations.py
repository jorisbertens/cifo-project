import numpy as np

def parametrized_ball_mutation(radius):
    '''
       For each weight in the network it generates a new weigth between coordinate - radius and coordinate - radius
    '''
    def ball_mutation(point, random_state):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation



def parametrized_ball_mutation_with_boundries(radius, search_space):
    '''
       For each weight in the network it generates a new weigth between coordinate - radius and coordinate - radius
       In case new value is outside the search space it is then reset to the boundary of the search space
    '''
    def ball_mutation(point, random_state):
        result = []
        for coordinate in point:
            mutated = random_state.uniform(low=coordinate - radius, high=coordinate + radius)
            while mutated < search_space[0] or mutated > search_space[1]:
                mutated = random_state.uniform(low=coordinate - radius, high=coordinate + radius)
            result.append(mutated)
        return np.array(result)
    return ball_mutation

def parametrized_random_member_mutation_fast(p, search_space):
    '''
        Selects a random subset of the weights with size: length of weights * p
        and replaces all the points with a single uniform number generated within the search space
    '''
    def random_member_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        new_points[indexes] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return random_member_mutation

def parametrized_random_member_mutation(p, search_space):
    '''
        Selects a random subset of the weights with size: length of weights * p
        and replaces all the points with uniform numbers generated within the search space

        https://www.researchgate.net/deref/http%3A%2F%2Fwww.ijcai.org%2FProceedings%2F89-1%2FPapers%2F122.pdf
    '''
    def random_member_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        for index in indexes:
            new_points[index] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return random_member_mutation

def parametrized_gaussian_mutation(p, mean, std):
    '''
        This operator adds a unit Gaussian distributed random value to the chosen genes.
        https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)
    '''
    def gaussian_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        for index in indexes:
            new_points[index] = random_state.normal(mean, std)
        return new_points
    return gaussian_mutation

def parametrized_swap_mutation(p):
    '''
        Randomly swaps length of weights * p values in the weights list
    '''
    def swap_mutation(point, random_state):
        size = int(( len(point) * p ) / 2)
        indexes1 = random_state.randint(low = 0,high = len(point), size = size)
        indexes2 = random_state.randint(low = 0,high = len(point), size = size)

        new_points = point.copy()

        new_points[indexes1], new_points[indexes2] = point[indexes2], point[indexes1]

        return new_points
    return swap_mutation

def parametrized_shrink_mutation(p, std):
    '''
    This operator adds a random number taken from a Gaussian distribution with mean equal to the original
    value of each decision variable characterizing the entry parent vector.

    https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)
    '''
    def shrink_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        for index in indexes:
            new_points[index] = random_state.normal(new_points[index], std)
        return new_points
    return shrink_mutation
