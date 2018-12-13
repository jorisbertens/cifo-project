import numpy as np

def phenotypic_entropy(population):
    '''
    measures the phenotypic entropy within the population (Vanneschi, 2018)
    '''
    length = len(population)
    fitnesses = ["{:.4f}".format(ind.fitness) for ind in population]
    unique_fitnesses, counts = np.unique(fitnesses, return_counts=True)
    return  np.sum([ count/length * np.log(count/length) for count in counts])


def phenotypic_variance(population):
    '''
    measures the phenotypic variance within the population (Vanneschi, 2018)
    '''
    average = np.mean([ind.fitness for ind in population])
    return np.sum([ np.power((indiv.fitness - average) , 2) for indiv in population]) / (len(population) - 1)


def genotypic_entropy(population):
    '''
    measures the genotypic entropy within a population (Vanneschi, 2018)
    '''
    length = len(population)
    individuals = [["{:.4f}".format(neuron) for neuron in indiv.representation ] for indiv in population]
    unique_fitnesses, counts = np.unique(individuals, return_counts=True)
    return np.sum([ count/length * np.log(count/length) for count in counts])


def genotypic_variance(population):
    '''
    measures the genotypic variance within a population (Vanneschi, 2018)
    '''
    avg_distance = np.sum(np.abs([ind.representation for ind in population]))
    return np.sum( [np.power(np.sum(np.abs((indiv.representation - avg_distance))),2) for indiv in population]) / (len(population) - 1)

