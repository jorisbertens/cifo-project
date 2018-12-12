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
from algorithms.pso import PSO


# setup logger
file_path =  "../../TestLog/other_algos/" + os.path.basename(__file__) + "_log.csv"
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


file_name= "../../LogFiles/" + os.path.basename(__file__) + "_log.csv"

header_string = "Seed,N_gen,Swarm_size,Social,Cognitive,Inertia,Pressure,Fitness,UnseenAccuracy,Time"
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

# PSO setup
seeds_per_run = [x for x in range(5)]

socials = [1.]
cognitves = [1.]
intertias = [.1]
n_gens = [100]


def algo_run(seed, n_gen, social ,cognitve, intertia):
    random_state = uls.get_random_state(seed)
    start_time = datetime.datetime.now()

    swarm_size = int(5000/n_gen)
    if swarm_size > 50:
        with open(file_name, "a") as myfile:
            print("Invalid parameters")
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
    alg = PSO(problem_instance=ann_op_i, random_state=random_state,
              swarm_size=swarm_size, social=social, cognitive=cognitve, inertia=intertia)

    alg.initialize()
    # initialize search algorithms
    ########Search   ############################ LOG \/ ########################
    alg.search(n_iterations=n_gen, report=True, log=True)

    ############# Evaluate unseen fitness ##################
    ann_i._set_weights(alg.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    time_elapsed = datetime.datetime.now() - start_time
    # Create result string
    result_string = ",".join(
        [str(seed), str(n_gen), str(swarm_size) ,str(swarm_size), str(social), str(cognitve), str(intertia),
         str(alg.best_solution.fitness), str(accuracy),str(time_elapsed)])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(header_string)
    print(result_string)

if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seeds_per_run,n_gens,socials,cognitves,intertias]))
    core_count = multiprocessing.cpu_count()
    print("All possible combinations generated:")
    print(possible_values)
    print("Number of cpu cores: "+str(core_count))

    ####### Magic appens here ########
    pool = multiprocessing.Pool(core_count-1)
    results = pool.starmap(algo_run, possible_values)
