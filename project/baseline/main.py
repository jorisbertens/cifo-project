import numpy as np
import utils as uls
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.genetic_algorithm import GeneticAlgorithm

# Hello
# Lukas Früchtnicht
# Alex
# Joris Bertens
# setup  random state
seed = 1
random_state = uls.get_random_state(seed)

#++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
#++++++++++++++++++++++++++
# import data
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

print(flat_images.shape)
print(digits.target_names)

# example
n_images = 25
plt.figure(figsize=(10, 10))
for i in range(n_images):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(digits.images[i], cmap=plt.cm.binary)
    plt.xlabel("Value: %d" % digits.target_names[digits.target[i]], fontsize=12)
plt.suptitle('Example of the training data',  fontsize=30)
plt.show()

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=random_state)

#++++++++++++++++++++++++++
# THE ANN
# restrictions:
# - 2 h.l. with sigmoid a.f.
# - softmax a.f. at output
# - 20% for validation
#++++++++++++++++++++++++++
# ann's ingridients
hl1 = 10
hl2 = 10
hidden_architecture = np.array([[hl1, sigmoid], [hl2, sigmoid]])
n_weights = X_train.shape[1]*hl1*hl2*len(digits.target_names)
validation_p = 0.2
# create ann
ann_i = ANN(hidden_architecture, softmax, accuracy_score,
                   (X_train, y_train), random_state, 0.2, digits.target_names)

#++++++++++++++++++++++++++
# THE PROBLEM INSTANCE
#++++++++++++++++++++++++++
validation_threshold = 0.1
ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                 minimization=False, validation_threshold=validation_threshold)