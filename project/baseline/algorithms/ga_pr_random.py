import logging
import numpy as np
from functools import reduce

from algorithms.random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithmProgressRateRandom(RandomSearch):
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

        for iteration in range(n_iterations):
            offsprings = []
            mutations_before = []
            mutations_after = []
            crossovers_before = []
            crossovers_after = []
            offsprings.extend([
                self._generate_random_valid_solution(),
                self._generate_random_valid_solution(),
                self._generate_random_valid_solution(),
            ])
            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in
                    range(2)]

                if self._random_state.uniform() < self.p_c:
                    crossovers_before.extend([off1, off2])
                    off1, off2 = self._crossover(p1, p2)
                    crossovers_after.extend([off1, off2])

                if self._random_state.uniform() < self.p_m:
                    mutations_before.extend([off1, off2])
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)
                    mutations_after.extend([off1, off2])

                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)

            offsprings.extend([elite])

            crossover_improvment = self.evaluate_improvment(crossovers_before,crossovers_after)
            mutation_improvment = self.evaluate_improvment(mutations_before,mutations_after)
            theta = self.convergence(offsprings)

            if crossover_improvment < mutation_improvment:
                self.p_m = self.p_m + theta
                self.p_c = self.p_c - theta
            else:
                self.p_m = self.p_m - theta
                self.p_c = self.p_c + theta

            print()
            print("Theta:" + str(theta))
            print("mutation_improvment:" + str(mutation_improvment) + " p_m: "+ str(self.p_m))
            print("crossover_improvment:" + str(crossover_improvment) + " p_c: "+ str(self.p_c))
            print("Elite: "+str(elite.fitness))

            if report:
                self._verbose_reporter_inner(elite, iteration)

            if log:
                log_event = [iteration, elite.fitness,
                             elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            self.population = offsprings

        self.best_solution = elite

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        self.problem_instance.evaluate(off1)
        self.problem_instance.evaluate(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        self.problem_instance.evaluate(mutant)
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

    def evaluate_improvment(self, before_pop, after_pop):
        fitness_sum_before = sum([indiv.fitness for indiv in before_pop])
        fitness_sum_after = sum([indiv.fitness for indiv in after_pop])
        if len(before_pop) == 0:
            return 0
        return (fitness_sum_after - fitness_sum_before) / len(before_pop)

    def convergence(self, pop):
        maximium = max([indiv.fitness for indiv in pop])
        average = np.mean([indiv.fitness for indiv in pop])
        if maximium == average:
            return 0.01

        minimum = min([indiv.fitness for indiv in pop])
        if maximium < minimum:
            return 0.01
        return 0.01 * ((maximium -average) / (maximium-minimum))