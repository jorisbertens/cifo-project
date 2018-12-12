import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
import crossovers as cross
import selections as sel
import mutations as mut

from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ga_2pop import GeneticAlgorithm2Pop
from algorithms.ga_dmr import GeneticAlgorithmDMR
from algorithms.ga_pr import GeneticAlgorithmProgressRate
from algorithms.ga_pr_random import GeneticAlgorithmProgressRateRandom
from algorithms.ga_mating_pool import GeneticAlgorithmMatingPool
from algorithms.ga_2pop_separate_c_m import GeneticAlgorithm2PopSeparateCM
from algorithms.ga_eval import GeneticAlgorithmEval
from algorithms.ga_elitism import GeneticAlgorithmElitism
from algorithms.ga_elitism_random import GeneticAlgorithmElitismRandom
from algorithms.ga_elitism_worst_removal import GeneticAlgorithmElitismWorstRemoval
from algorithms.ga_2pop_random import GeneticAlgorithm2Random
from algorithms.ga_dc import GeneticAlgorithmDeterministicCrowding
from algorithms.ga_2_pop_dc import GeneticAlgorithm2PopDeterministicCrowding


import sys
import os
from subprocess import call
0
# setup logger
file_path =  "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv")
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


file_name= "LogFiles/" + "custom_file" + str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"

header_string = "Fitness,UnseenAccuracy,Seed,N_gen,PS,control,update_rate,radius,time"
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
seeds_per_run = [0,1,2,3,4]
n_genes = [200,300]
controls = [1,2,3]
update_rates = [0.7, 0.8, 0.9]
radiuses = [0.003, 0.001]


def algo_run(seed, n_gen, control, update_rate, radius):
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
    mut_algo = mut.parametrized_random_member_mutation(radius, [-4,4])
    alg = SimulatedAnnealing(ann_op_i, random_state, pop_size, mut_algo, control, update_rate)
    alg.initialize()
    # initialize search algorithms
    ########Search   ############################ LOG \/ ########################
    alg.search(n_iterations=n_gen, report=False, log=False)

    ############# Evaluate unseen fitness ##################
    ann_i._set_weights(alg.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    time_elapsed = datetime.datetime.now() - start_time
    # Create result string
    result_string = ",".join(
        [str(alg.best_solution.fitness), str(accuracy),
         str(seed), str(n_gen), str(pop_size),
         str(control), str(update_rate), str(radius), str(time_elapsed)
         ])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(result_string)
    if alg.best_solution.fitness > 0.7 and alg.best_solution.valid:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!yey!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seeds_per_run,n_genes,controls, update_rates, radiuses]))
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
