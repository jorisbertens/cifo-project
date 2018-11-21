import numpy as np

def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation

def parametrized_ball_mutation2(radius, search_space):
    def ball_mutation(point, random_state):
        result = []
        for coordinate in point:
            mutated = random_state.uniform(low=coordinate - radius, high=coordinate + radius)
            while mutated < search_space[0] or mutated > search_space[1]:
                mutated = random_state.uniform(low=coordinate - radius, high=coordinate + radius)
            result.append(mutated)
        return np.array(result)
    return ball_mutation

def parametrized_random_member_mutation(p, search_space):
    def SA_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        new_points[indexes] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return SA_mutation

def parametrized_random_member_mutation2(p, search_space):
    def SA_mutation(point, random_state):

        indexes = random_state.randint(low = 0,high = len(point), size = int(len(point) * p ))
        new_points = point.copy()
        for index in indexes:
            new_points[index] = random_state.uniform(low=search_space[0], high=search_space[1])
        return new_points
    return SA_mutation

