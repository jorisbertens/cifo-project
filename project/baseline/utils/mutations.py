import numpy as np

def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation

def parametrized_ball_mutation_with_boundries(radius, search_space):
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
    def random_member_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        new_points[indexes] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return random_member_mutation

def parametrized_random_member_mutation(p, search_space):
    '''
        https://www.researchgate.net/deref/http%3A%2F%2Fwww.ijcai.org%2FProceedings%2F89-1%2FPapers%2F122.pdf
    '''
    def random_member_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        for index in indexes:
            new_points[index] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return random_member_mutation
def parametrized_biased_random_member_mutation(p, search_space):
    return 0

def parametrized_swap_mutation(p):
    def swap_mutation(point, random_state):
        size = int(( len(point) * p ) / 2)
        indexes1 = random_state.randint(low = 0,high = len(point), size = size)
        indexes2 = random_state.randint(low = 0,high = len(point), size = size)

        new_points = point.copy()

        new_points[indexes1], new_points[indexes2] = point[indexes2], point[indexes1]

        return new_points
    return swap_mutation

def parameterized_random_mutation(ball_radius, random_member_p, swap_p, search_space):
    def random_mutation(point, random_state):
        mutations = [
            parametrized_ball_mutation(ball_radius),
            parametrized_ball_mutation_with_boundries(ball_radius, search_space),
            parametrized_random_member_mutation_fast(random_member_p, search_space),
            parametrized_random_member_mutation(random_member_p, search_space),
            parametrized_swap_mutation(swap_p)
        ]
        mutation_algo = random_state.choice(mutations)
        return mutation_algo(point, random_state)
    return random_mutation