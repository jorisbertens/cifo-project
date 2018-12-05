import random

# roulette wheel
def fitness_proportionate_selection(population):
    def roulette_wheel_selection(random_state):
        roulette_wheel_selection = []
        for j in range(int(erate * population_size)):
            max = sum(solution.fitness for solution in population)
            pick = random.uniform(0, max)
            i = 0
            counter = population[0].fitness
            while counter < rval and i < len(population) - 1:
                counter += population[i].fitness
                i += 1
        roulette_wheel_selection.append(population[i])

    return roulette_wheel_selection

# ranking
import sys

def compute_sel_prob(population.fitness):
    n = len(population.fitness)
    rank_sum = n * (n + 1) / 2
    for rank, solution.fitness in enumerate(sorted(population.fitness), 1):
        yield rank, ind_fitness, float(rank) / rank_sum


