import logging
import numpy as np
from functools import reduce

from algorithms.random_search import RandomSearch
from solutions.solution import Solution
import mutations as mut


class GeneticAlgorithmGrowPop(RandomSearch):
    '''
    Starting with a population of two and growing till a population of the maximum population value (5000/n_gen),
    where the population grows by one over every generation.
    It was based on the on-the-fly population size adjustment discussed in Eiben et al. (2018)
    '''
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = 2
        self.max_pop_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False, log=False):
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
        diversities=[]

        while len(self.population) < self.max_pop_size:
            offsprings = []

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            offsprings.extend(self._get_x_elites(self.population, 1))

            self.population = offsprings

        elite = self.best_solution
        pop_size=len(self.population)
        n_iterations = int(int((5000 - pop_size*(pop_size+1)/2 )) // pop_size)
        print("Done:" + str(n_iterations))
        for iteration in range(n_iterations):
            offsprings = []

            while len(offsprings) < pop_size:
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)


                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])

            while len(offsprings) > pop_size:
                offsprings.pop()

            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)
            diversities.append(self._phenotypic_diversity_shift(offsprings))
            print(str(iteration))
            print("Fitness: "+str(elite.fitness))
            print("Diversity: "+str(sum(diversities) / len(diversities)))
            print()

            if report:
                self._verbose_reporter_inner(elite, iteration)

            if log:
                log_event = [iteration, elite.fitness, elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            self.population = offsprings

        self.best_solution = elite

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2


    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def _get_x_elites(self, population,x):
        return sorted(population, key=lambda x: x.fitness, reverse=not self.problem_instance.minimization)[:x]