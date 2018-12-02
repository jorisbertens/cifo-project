import logging
import numpy as np
from functools import reduce

from algorithms.random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithm2PopDeterministicCrowding(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m, elite_count):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.elite_count = elite_count

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

        pop_size = len(self.population1)
        for iteration in range(n_iterations):
            offsprings1 = []
            offsprings2 = []
            while len(offsprings1) < pop_size:
                off1, off2 = p1, p2 = [
                    self.selection(self.population1, self.problem_instance.minimization, self._random_state) for _ in
                    range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    offsprings1.extend([off1, off2])

                if (self._euclidian_distance(p1.representation,off1.representation) + self._euclidian_distance(p2.representation,off2.representation)) <= (
                        self._euclidian_distance(p1.representation, off2.representation) + self._euclidian_distance(p2.representation, off1.representation)):
                    offsprings1.extend([off1]) if off1.fitness > p1.fitness else offsprings1.extend([p1])
                    offsprings1.extend([off2]) if off2.fitness > p2.fitness else offsprings1.extend([p2])
                else:
                    offsprings1.extend([off1]) if off1.fitness > p2.fitness else offsprings1.extend([p2])
                    offsprings1.extend([off1]) if off1.fitness > p2.fitness else offsprings1.extend([p2])

            while len(offsprings1) > pop_size:
                offsprings1.pop()

            while len(offsprings2) < pop_size:
                off1, off2 = p1, p2 = [
                    self.selection(self.population2, self.problem_instance.minimization, self._random_state) for _ in
                    range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    offsprings2.extend([off1, off2])

                if (self._euclidian_distance(p1.representation,off1.representation) + self._euclidian_distance(p2.representation,off2.representation)) <= (
                        self._euclidian_distance(p1.representation, off2.representation) + self._euclidian_distance(p2.representation, off1.representation)):
                    offsprings2.extend([off1]) if off1.fitness > p1.fitness else offsprings2.extend([p1])
                    offsprings2.extend([off2]) if off2.fitness > p2.fitness else offsprings2.extend([p2])
                else:
                    offsprings2.extend([off1]) if off1.fitness > p2.fitness else offsprings2.extend([p2])
                    offsprings2.extend([off1]) if off1.fitness > p2.fitness else offsprings2.extend([p2])

            while len(offsprings2) > pop_size:
                offsprings2.pop()

            elite_offspring1 = self._get_elite(offsprings1)
            elite1 = self._get_best(elite1, elite_offspring1)

            elite_offspring2 = self._get_elite(offsprings2)
            elite2 = self._get_best(elite2, elite_offspring2)

            offsprings1.extend(self._get_x_elites(self.population2, self.elite_count))
            offsprings2.extend(self._get_x_elites(self.population1, self.elite_count))

            if report:
                self._verbose_reporter_inner(elite1, iteration)

            if log:
                log_event = [iteration, elite1.fitness,
                             elite1.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m,
                             self._phenotypic_diversity_shift(offsprings1)]
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
        return np.std(fitness_offsprings) - np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def _generate_random_valid_solutions_with_size(self, size):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(size)])
        return solutions

    def _get_x_elites(self, population, x):
        return sorted(population, key=lambda x: x.fitness, reverse=not self.problem_instance.minimization)[:x]

    def _euclidian_distance(self, x,y):
        return np.sqrt(np.sum((x - y) ** 2))