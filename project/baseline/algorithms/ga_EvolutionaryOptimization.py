import logging
import numpy as np
from functools import reduce

from random_search import RandomSearch
from solution import Solution


class EvolutionaryOptimization(RandomSearch):
    '''
        https://arxiv.org/abs/1806.09819v1
    '''
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
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

        elite = self.best_solution
        pop_size = len(self.population)
        for iteration in range(n_iterations):
            offsprings= []
            elite_offsprings = []
            mutation_offsprings = []
            crossover_offsprings = []

            while len(crossover_offsprings) < (pop_size * self.p_c):
                p1, p2 = [
                        self.selection(self._get_x_elites(self.population, int(pop_size * 0.5)),
                        self.problem_instance.minimization,
                        self._random_state) for _ in range(2)
                ]

                off1, off2 = self._crossover(p1, p2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                crossover_offsprings.extend([off1, off2])

            while len(crossover_offsprings) < (pop_size * self.p_c):
                crossover_offsprings.pop()

            while len(mutation_offsprings) < (pop_size * self.p_m):
                p1= self.selection(self._get_x_elites(self.population, int(pop_size * 0.5)),
                        self.problem_instance.minimization,
                        self._random_state)


                off1= self._mutation(p1)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                mutation_offsprings.extend([off1])

            while len(mutation_offsprings) < (pop_size * self.p_m):
                mutation_offsprings.pop()

            elite_offsprings = self._get_x_elites(self.population, int(pop_size * (1 - (self.p_c + self.p_m))))

            print("Elite count: "+str(len(elite_offsprings))+" Elite Fitness:"+ str(self._get_elite(elite_offsprings).fitness))
            print("Mut count: "+str(len(mutation_offsprings))+" Mut Fitness:"+ str(self._get_elite(mutation_offsprings).fitness))
            print("Cross count: "+str(len(crossover_offsprings))+" Cross Fitness:"+ str(self._get_elite(crossover_offsprings).fitness))
            print()
            offsprings = np.concatenate([elite_offsprings, mutation_offsprings, crossover_offsprings])

            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)

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