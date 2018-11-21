import numpy as np
from functools import reduce

def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection

def random_selection(population, minimization, random_state):
    return random_state.choice(population)

def best_selection(population, minimization, random_state):
    return sorted(population, key=lambda x: x.fitness, reverse=not minimization)[0]

def roulette_selection(population, minimization, random_state):
    ## MINIMIZATION (NOT WELL IMPLEMENTED)
    max_fitness = max(ind.fitness for ind in population)
    sum_fits = sum(max_fitness - ind.fitness for ind in population)
    pick = random_state.uniform(0, sum_fits)
    current = 0
    for ind in population:
        current += ind.fitness
        if current > pick:
            return ind


def boltzmann_selection(pressure, n_gen):
    gen = 0
    def tournament_selection(population, minimization, random_state):
        nonlocal  gen
        factor = pressure - ((pressure / 2) * (gen/(n_gen*len(population))))
        gen = gen + 1
        tournament_pool_size = int(len(population)* factor)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection