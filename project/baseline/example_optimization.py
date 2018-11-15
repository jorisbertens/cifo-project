from main import *


#++++++++++++++++++++++++++
# THE OPTIMIZATION
#++++++++++++++++++++++++++
n_gen = 50
ps = 100
p_c = 0.50
p_m = 0.9
radius = 0.2
size=0.1

ga = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(size),
                      uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_c)
ga.initialize()
ga.search(n_gen, True)

ga.best_solution.print_()
print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)

#++++++++++++++++++++++++++
# VISUALIZE FIT
#++++++++++++++++++++++++++
x1_start, x1_stop = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_start, x2_stop = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

step = 0.01

x1_dim = np.arange(x1_start, x1_stop, step)
x2_dim = np.arange(x2_start, x2_stop, step)

xx, yy = np.meshgrid(x1_dim, x2_dim)
grid = np.c_[xx.ravel(), yy.ravel()]

ann_i._set_weights(ga.best_solution.representation)
Z = ann_i.stimulate_with(grid)
Z = np.where(Z > .5, 1, 0)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=100, edgecolor="black", linewidths=1)
plt.show()