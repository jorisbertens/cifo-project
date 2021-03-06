import numpy as np
from functools import reduce
import operator

# Selection algorithms

def parametrized_tournament_selection(pressure):
    '''
    Chooses random individuals into a tournament pool,
    where the fittest individual will be selected to be a parent (Mitchell, 1999; Jebari & Madiadi, 2013)
    '''
    def tournament_selection(population, minimization, random_state, fitness_sharing=False):
        fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'

        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return min(tournament_pool, key=operator.attrgetter(fitness_name))
        else:
            return max(tournament_pool, key=operator.attrgetter(fitness_name))

    return tournament_selection

def random_selection(population, minimization, random_state, fitness_sharing=False):
    '''
    Chooses random individuals of the population to be parents
    '''
    return random_state.choice(population)

def best_selection(population, minimization, random_state, fitness_sharing=False):
    '''
    Returns as parents the elite in the population. It perverts the concept of Genetic algorithms and  approximates to a neighborhood based algorithm
    '''
    fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'
    return sorted(population, key=operator.attrgetter(fitness_name), reverse=not minimization)[0]

def parameterized_x_best_selection(num):
    '''
    selects one individual among the best x ones
    '''
    def best_x_selection(population, minimization, random_state, fitness_sharing=False):
        fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'
        return random_state.choice(sorted(population, key=operator.attrgetter(fitness_name), reverse=not minimization)[:num])
    return best_x_selection

def parameterized_best_or_random_selection(p):
    '''
    selects the best or a random individual based on a given probability
    '''
    def best_or_random_selection(population, minimization, random_state, fitness_sharing=False):
        if random_state.uniform(0,1) < p:
            fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'
            return sorted(population, key=operator.attrgetter(fitness_name), reverse=not minimization)[0]
        return random_state.choice(population)
    return best_or_random_selection

def rank_selection(population, minimization, random_state, fitness_sharing=False):
    '''
    Calculates a new fitness value based on their rank comparing with the others and performs a roulette-wheel-selection after (Mitchell, 1999)
    '''
    fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'
    sorted_pop = sorted(population, key=operator.attrgetter(fitness_name), reverse=minimization)
    length = len(sorted_pop)
    gaussian_sum =  ((length-1) * length) / 2
    pick = random_state.uniform(0, gaussian_sum)
    current = 0
    for i, ind in enumerate(sorted_pop):
        current += i
        if current >= pick:
            return ind


def stochastic_universal_sampling():
    '''
    not implemented due to time issues
    '''
    return 0

def roulette_selection(population, minimization, random_state, fitness_sharing=False):
    '''
    Selects individuals to be parents based on their fitness proportion (Mitchell, 1999; Jebari & Madiadi, 2013)
    '''
    fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'

    sorted_pop = sorted(population, key=operator.attrgetter(fitness_name), reverse=not minimization)

    max_fitness = max(population, key=operator.attrgetter(fitness_name))
    if minimization:
        sum_fits = sum(max_fitness - operator.attrgetter(fitness_name)(ind) for ind in population)
    else:
        sum_fits = sum(operator.attrgetter(fitness_name)(ind) for ind in population)

    pick = random_state.uniform(0, sum_fits)
    current = 0
    for ind in sorted_pop:
        if minimization:
            current += (max_fitness - ind.fitness)
        else:
            current += ind.fitness

        if current > pick:
            return ind


def increasing_tournament_size_selection(pressure, n_gen):
    '''
    Selection pressure is gradually increasing over the number of generations,
    which means that in the beginning the probability for every individual to be chosen is high and lowers with the number of generations.
    Inspired by Boltzmann selection (Mitchell, 1999)
    '''
    gen = 0
    def tournament_selection(population, minimization, random_state, fitness_sharing=False):
        nonlocal  gen
        fitness_name = "fitness" if not fitness_sharing else 'custom_fitness'

        factor = pressure - ((pressure / 2) * (gen/(n_gen* len(population))))
        gen = gen + 1
        tournament_pool_size = int(len(population) * factor)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return min(tournament_pool, key=operator.attrgetter(fitness_name))
        else:
            return max(tournament_pool, key=operator.attrgetter(fitness_name))

    return tournament_selection