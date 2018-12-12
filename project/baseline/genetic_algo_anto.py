import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils.utils as uls
import utils.crossovers as cross
import utils.selections as sel
import utils.mutations as mut

from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ga_2pop import GeneticAlgorithm2Pop
from algorithms.ga_dmr import GeneticAlgorithmDMR
from algorithms.ga_pr import GeneticAlgorithmProgressRate
from algorithms.ga_mating_pool import GeneticAlgorithmMatingPool
from algorithms.ga_2pop_separate_c_m import GeneticAlgorithm2PopSeparateCM
from algorithms.ga_eval import GeneticAlgorithmEval
from algorithms.ga_elitism import GeneticAlgorithmElitism
from algorithms.ga_elitism_random import GeneticAlgorithmElitismRandom
from algorithms.ga_elitism_worst_removal import GeneticAlgorithmElitismWorstRemoval
from algorithms.ga_dc import GeneticAlgorithmDeterministicCrowding
from algorithms.ga_drop_worst import GeneticAlgorithmDropWorst
from algorithms.ga_growpop_elitism import GeneticAlgorithmGrowPopElitism
from algorithms.ga_single_elite_start import GeneticAlgorithmSingleEliteStart
from algorithms.ga_fitness_sharing import GeneticAlgorithmFitnessSharing
# setup logger
file_path =  "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv")
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


file_name= "LogFiles/" + "custom_file" + str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"

header_string = "Fitness,UnseenAccuracy,Seed,N_gen,PS,PC,PM,radius,Pressure,elite_count,Time,alg,sel,cross,mut"
with open(file_name, "a") as myfile:
    myfile.write(header_string + "\n")


# ++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
# ++++++++++++++++++++++++++
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=0)

# setup benchmarks
validation_p = .2
validation_threshold = .07

# Genetic Algorithm setup
seeds_per_run = [2]
n_genes = [100,200,300,500]
p_cs = [0.5,0.7,0.9]
p_ms = [0.5,0.7,0.9]
radiuses= [0.07,0.05,0.03,0.01]
pressures = [0.5,1,2]
elite_counts = [3]

#0.6268191268191268,0.5454545454545454,2,100,50,1,1,3,1,3,0:03:47.689362,<algorithms.ga_elitism_worst_removal.GeneticAlgorithmElitismWorstRemoval object at 0x7f8fc59a0fd0>,<function boltzmann_selection.<locals>.tournament_selection at 0x7f8fe80ca0d0>,<function parametrized_two_point_crossover.<locals>.two_point_crossover at 0x7f8fc5c50bf8>,<function parametrized_random_member_mutation.<locals>.random_member_mutation at 0x7f8fc5c50c80>

def algo_run(seed, n_gen, p_c, p_m, radius, pressure, elite_count):
    random_state = uls.get_random_state(seed)
    start_time = datetime.datetime.now()

    pop_size = int(5000/n_gen)
    if pop_size > 50:
        with open(file_name, "a") as myfile:
            myfile.write("Invalid parameters" + "\n")
        return 0

    #++++++++++++++++++++++++++
    # THE ANN
    # restrictions:
    # - 2 h.l.
    # - Softmax a.f. at output
    # - 20%, out of remaining 67%, for validation
    #++++++++++++++++++++++++++
    # ann's architecture
    hidden_architecture = np.array([[10, sigmoid], [10, sigmoid]])
    n_weights = X_train.shape[1]*10*10*len(digits.target_names)
    # create ann
    ann_i = ANN(hidden_architecture, softmax, accuracy_score, (X_train, y_train), random_state, validation_p, digits.target_names)

    #++++++++++++++++++++++++++
    # THE PROBLEM INSTANCE
    # - optimization of ANN's weights is a COP
    #++++++++++++++++++++++++++
    ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                     minimization=False, validation_threshold=validation_threshold)

    #++++++++++++++++++++++++++
    # THE SEARCH
    # restrictions:
    # - 5000 offsprings/run max*
    # - 50 offsprings/generation max*
    # - use at least 5 runs for your benchmarks
    # * including reproduction
    #++++++++++++++++++++++++++
    sel_algo = sel.boltzmann_selection(1, n_gen)
    cross_algo = cross.two_point_crossover
    mut_algo = mut.parametrized_random_member_gaussian_ball_mutation(pressure, radius)

    alg = GeneticAlgorithmFitnessSharing(ann_op_i, random_state, pop_size, sel_algo,
                      cross_algo, p_c, mut_algo, p_m)
    alg.initialize()
    # initialize search algorithms
    ########Search   ############################ LOG \/ ########################
    alg.search(n_iterations=n_gen, report=False, log=True)

    ############# Evaluate unseen fitness ##################
    ann_i._set_weights(alg.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    time_elapsed = datetime.datetime.now() - start_time
    # Create result string
    result_string = ",".join(
        [str(alg.best_solution.fitness), str(accuracy),
         str(seed), str(n_gen), str(pop_size),
         str(p_c), str(p_m), str(radius), str(pressure), str(elite_count),
         str(time_elapsed),
         str(alg), str(sel_algo), str(cross_algo), str(mut_algo)
         ])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(result_string)
    if alg.best_solution.fitness > 0.7:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!yey!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seeds_per_run,n_genes,p_cs,p_ms,radiuses,pressures,elite_counts]))
    core_count = multiprocessing.cpu_count()
    print("All possible combinations generated:")
    print(possible_values)
    print(len(possible_values))
    print("Number of cpu cores: "+str(core_count))
    print()
    print(header_string)

    ####### Magic appens here ########
    pool = multiprocessing.Pool(core_count-1)
    results = pool.starmap(algo_run, possible_values)
