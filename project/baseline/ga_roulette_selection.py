from main import *

#++++++++++++++++++++++++++
# THE OPTIMIZATION
# restrictions:
# - 5000 f.e./run
# - 50 f.e./generation
# - use at least 5 runs for benchmarks
#++++++++++++++++++++++++++
n_gen = 10
ps = 20
p_c = .5
p_m = .9
radius = .2
pressure = .2
ga = GeneticAlgorithm(ann_op_i, random_state, ps, sel.boltzmann_selection(0.4,n_gen),
                      cross.one_point_crossover, p_c, mut.parametrized_ball_mutation(radius), p_m)
ga.initialize()
ga.search(n_gen, False, True)

ga.best_solution.print_()
print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % ga.best_solution.validation_fitness)

sa = SimulatedAnnealing(ann_op_i, random_state, ps, mut.parametrized_ball_mutation(radius), 2, .9)
sa.initialize()
sa.search(n_gen, True, True)

sa.best_solution.print_()
print("Training fitness of the best solution: %.2f" % sa.best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % sa.best_solution.validation_fitness)
