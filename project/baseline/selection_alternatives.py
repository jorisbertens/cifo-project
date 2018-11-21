import random

def fitness_proportionate_selection(population):
    def roulette_wheel_selection(minimization, random_state):
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


