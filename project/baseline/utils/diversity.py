import numpy as np


def phenotypic_entropy(population):
    length = len(population)
    fitnesses = [str(ind.fitness)[5] for ind in population]
    unique_fitnesses, counts = np.unique(fitnesses, return_counts=True)
    return  np.sum([ count/length * np.log(count/length) for count  in counts])


def phenotypic_variance(population):
    average = np.mean([ind.fitness for ind in population])
    return np.sum([ np.power((indiv.fitness - average),2) for indiv in population]) / (len(population) - 1)


def genotypic_entropy(population):
    length = len(population)
    individuals = [[str(neuron)[4] for neuron in indiv.representation ] for indiv in population]
    unique_fitnesses, counts = np.unique(individuals, return_counts=True)
    return  np.sum([ count/length * np.log(count/length) for count  in counts])


def genotypic_variance(population):
    avg_distance = np.sum(np.abs([ind.representation for ind in population]))
    return np.sum( [np.power((indiv.representation - avg_distance),2) for indiv in population]) / (len(population) - 1)

