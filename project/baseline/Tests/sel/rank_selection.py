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

from ANNOP import ANNOP
from ANN import ANN, softmax, sigmoid
from genetic_algorithm import GeneticAlgorithm
from ga_2pop import GeneticAlgorithm2Pop
from ga_dmr import GeneticAlgorithmDMR
from ga_pr import GeneticAlgorithmProgressRate
from ga_pr_random import GeneticAlgorithmProgressRateRandom
from ga_mating_pool import GeneticAlgorithmMatingPool
from ga_2pop_separate_c_m import GeneticAlgorithm2PopSeparateCM
from ga_eval import GeneticAlgorithmEval
from ga_elitism import GeneticAlgorithmElitism
from ga_elitism_random import GeneticAlgorithmElitismRandom
from ga_elitism_worst_removal import GeneticAlgorithmElitismWorstRemoval
from ga_2pop_random import GeneticAlgorithm2Random
from ga_dc import GeneticAlgorithmDeterministicCrowding
from ga_drop_worst import GeneticAlgorithmDropWorst
from ga_growpop import GeneticAlgorithmGrowPop
from ga_single_elite_start import GeneticAlgorithmSingleEliteStart

# setup logger
# !!!!!!!!!!!!!!!!!!!!!Change file name !!!!!!!!!!!!!!!!!!!!!!!!!!!1
file_path =  "../../TestLog/" + os.path.basename(__file__) + "_log.csv"
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


file_name= "../../LogFiles/" + os.path.basename(__file__) + "_log.csv"

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
''' PLEASE CHECK SETUP!!!!! '''


# Genetic Algorithm setup
# !!!!!!!!!!!!!!!!!!! Baseline parameters !!!!!!!!!!!!!!!!!!!
seeds_per_run = [0,1,2,3,4]
n_genes = [180]#is fixed
p_cs = [1]#is fixed
p_ms = [0.5]#is fixed
radiuses= [0.6]#is fixed
pressures = [0.8]                  ###change
elite_counts = [0]#is na

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
    #!!!!!!!!!!!!!!!!!!!!!!!!! Baseline Parameters !!!!!!!!!!!!!!!!!!!
    sel_algo = sel.rank_selection
    cross_algo = cross.one_point_crossover
    mut_algo = mut.parametrized_ball_mutation(radius)

    alg = GeneticAlgorithm(ann_op_i, random_state, pop_size, sel_algo,
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
    pool = multiprocessing.Pool(2)
    results = pool.starmap(algo_run, possible_values)
