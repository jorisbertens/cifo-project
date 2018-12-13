import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np
import pickle

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
import crossovers as cross
import selections as sel
import mutations as mut

from ANNOP import ANNOP
from ANN import ANN, softmax, sigmoid
from ga_fitness_sharing import GeneticAlgorithmFitnessSharing

# setup logger
file_path =  "best_solution_log.csv"
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

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
seeds_per_run = [11]
n_genes = [312]
p_cs = [1]
p_ms = [1]
radiuses= [0.013]
pressures = [1]
elite_counts = [1]
std = [2.81]

def save_object(representation, fullpath):
    with open(fullpath, 'wb') as output:
        pickle.dump(representation, output, pickle.HIGHEST_PROTOCOL)


def algo_run(seed, n_gen, p_c, p_m, radius, pressure, elite_count):
    random_state = uls.get_random_state(seed)
    start_time = datetime.datetime.now()

    pop_size = int(5000/n_gen)
    if pop_size > 50:
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
    sel_algo = sel.increasing_tournament_size_selection(pressure, n_gen)
    cross_algo = cross.geometric_crossover
    mut_algo = mut.parametrized_shrink_mutation(radius, std)

    alg = GeneticAlgorithmFitnessSharing(ann_op_i, random_state, pop_size, sel_algo,
                      cross_algo, p_c, mut_algo, p_m)
    alg.initialize()
    # initialize search algorithms
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
         str(p_c), str(p_m), str(radius), str(pressure), str(elite_count),
         str(time_elapsed),
         str(alg), str(sel_algo), str(cross_algo), str(mut_algo)
         ])
    # Output very good result to terminal
    print(accuracy)
    save_object(alg.best_solution, 'representation.pkl')


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seeds_per_run,n_genes,p_cs,p_ms,radiuses,pressures,elite_counts]))
    core_count = multiprocessing.cpu_count()
    print("All possible combinations generated:")
    print(possible_values)
    print(len(possible_values))
    print("Number of cpu cores: "+str(core_count))
    print()
    ####### Magic appens here ########
    pool = multiprocessing.Pool(core_count)
    results = pool.starmap(algo_run, possible_values)
