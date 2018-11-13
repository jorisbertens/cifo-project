from main import *
import itertools
from datetime import datetime

#++++++++++++++++++++++++++
# THE OPTIMIZATION
#++++++++++++++++++++++++++

n_gen = 50
ps = 100
p_c = 0.50
p_m = 0.9
radius = 0.2
size=0.1

result_string = ",".join(["Test run",str(n_gen),str(ps),str(p_c),str(p_m),str(radius),str(size),"0"])

with open("antonio_tests.csv", "a") as myfile:
    myfile.write(result_string)


sizes = [ x*0.1 for x in range(1,5) ]
radiuses = [ x*0.2 for x in range(1,5) ]
p_ms = [x * 0.1 for x in range(5, 10) ]

possible_values = list(itertools.product(*[sizes,radiuses,p_ms]))
counter = 0
for size,radius,p_m in possible_values:
    start_time = datetime.now()

    run_results = []
    for seed in range(0, 5):
        algo_start_time = datetime.now()
        #setup random state
        random_state = uls.get_random_state(seed)

        ga = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(size),
                              uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)
        ga.initialize()
        ga.search(n_gen, False)
        print("Counter: "+str(counter)+" Seed:"+str(seed)+ " Time elapsed:"+str(datetime.now() - algo_start_time))
        ga.best_solution.print_()
        print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)

        run_results.append(ga.best_solution.fitness)

    time_elapsed = datetime.now() - start_time
    print("Run results: "+str(np.mean(run_results)))
    print(run_results)
    result_string = ";".join(["Run:"+str(counter),str(n_gen),str(ps),str(p_c),str(p_m),str(radius),str(size),str(np.mean(run_results)),str(time_elapsed)])
    with open("antonio_tests.csv", "a") as myfile:
        myfile.write(result_string + "\n")
    counter+=1


# analyze data-frame
print("Results:\n"+str(run_results))
print("Descriptive statistics:\n"+str(results.describe()))

