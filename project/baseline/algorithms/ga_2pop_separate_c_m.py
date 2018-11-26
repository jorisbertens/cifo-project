import logging
import numpy as np
from functools import reduce

from algorithms.random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithm2PopSeparateCM(RandomSearch):
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
        self.population1 = self._generate_random_valid_solutions_with_size(self.population_size // 2)
        self.population2 = self._generate_random_valid_solutions_with_size(self.population_size // 2)

        elite1 = self._get_elite(self.population1)
        elite2 = self._get_elite(self.population2)
        self.best_solution = self._get_best(elite1, elite2)

    def search(self, n_iterations, report=False, log=False):
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite1 = self._get_elite(self.population1)
        elite2 = self._get_elite(self.population2)

        for iteration in range(n_iterations):
            offsprings1 = []
            offsprings2 = []

            while len(offsprings1) < len(self.population1):
                off1, off2 = p1, p2 = [
                    self.selection(self.population1, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                off1, off2 = self._crossover(p1, p2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    offsprings1.extend([off1, off2])

            while len(offsprings1) > len(self.population1):
                offsprings1.pop()

            while len(offsprings2) < len(self.population2):
                off1, off2 = p1, p2 = [
                    self.selection(self.population2, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                off1 = self._mutation(off1)
                off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    offsprings2.extend([off1, off2])

            while len(offsprings2) > len(self.population2):
                offsprings2.pop()

            elite_offspring1 = self._get_elite(offsprings1)
            elite1 = self._get_best(elite1, elite_offspring1)

            elite_offspring2 = self._get_elite(offsprings2)
            elite2 = self._get_best(elite2, elite_offspring2)

            offsprings1.extend([elite2, elite2])
            offsprings2.extend([elite1, elite1])

            if report:
                self._verbose_reporter_inner(elite, iteration)

            if log:
                log_event = [iteration, elite.fitness, elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            self.population1 = offsprings1
            self.population2 = offsprings2

        elite1 = self._get_elite(self.population1)
        elite2 = self._get_elite(self.population2)
        self.best_solution = self._get_best(elite1, elite2)

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

    def _generate_random_valid_solutions_with_size(self, size):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(size)])
        return solutions