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
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.random_search import RandomSearch


# setup logger
file_path =  "LogFiles/" + "random.csv"
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')


file_name= "LogFiles/random.csv"

header_string = "Seed,Fitness,UnseenAccuracy,Time"
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


n_gen = 100
validation_p = .2
validation_threshold = .07
seed=1610
while True:
    random_state = uls.get_random_state(seed)

    hidden_architecture = np.array([[10, sigmoid], [10, sigmoid]])
    n_weights = X_train.shape[1] * 10 * 10 * len(digits.target_names)
    ann_i = ANN(hidden_architecture, softmax, accuracy_score, (X_train, y_train), random_state, validation_p,
                digits.target_names)
    ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                     minimization=False, validation_threshold=validation_threshold)

    # ++++++++++++++++++++++++++
    # THE SEARCH
    # restrictions:
    # - 5000 offsprings/run max*
    # - 50 offsprings/generation max*
    # - use at least 5 runs for your benchmarks
    # * including reproduction
    # ++++++++++++++++++++++++++
    start_time = datetime.datetime.now()

    alg = RandomSearch(ann_op_i, random_state )
    alg.initialize()
    # initialize search algorithms
    ########Search   ############################ LOG \/ ########################
    alg.search(n_iterations=5000, report=False, log=False)

    ############# Evaluate unseen fitness ##################
    ann_i._set_weights(alg.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    time_elapsed = datetime.datetime.now() - start_time
    # Create result string
    result_string = ",".join(
        [str(seed),str(alg.best_solution.fitness), str(accuracy), str(time_elapsed)])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(header_string)
    print(result_string)
    seed += 1