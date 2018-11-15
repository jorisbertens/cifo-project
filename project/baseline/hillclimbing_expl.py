from main import *
from algorithms.hill_climbing import HillClimbing, RandomSearch
import itertools
from datetime import datetime

#++++++++++++++++++++++++++
# THE OPTIMIZATION
#++++++++++++++++++++++++++
neighborhood_size = 100
n_gen = 50
#hc = HillClimbing(problem_instance=ann_op_i, random_state=random_state, neighborhood_size=neighborhood_size, neighborhood_function=uls.bit_flip)


result_string = ",".join(["Test run",str(n_gen),str(neighborhood_size),"0"])

with open("joris_tests.csv", "a") as myfile:
    myfile.write(result_string)


neighborhood_size = [ x for x in range(100,1000,50) ]
n_gens = [ x for x in range(50,300,50) ]
methods = [uls.bit_flip]
possible_values = list(itertools.product(*[neighborhood_size,n_gens, methods]))
print(possible_values)
counter = 0
for neighborhood_size,n_gen,method in possible_values:
    start_time = datetime.now()

    run_results = []
    for seed in range(0, 5):
        algo_start_time = datetime.now()
        #setup random state
        random_state = uls.get_random_state(seed)

        hc = HillClimbing(problem_instance=ann_op_i, random_state=random_state, neighborhood_size=neighborhood_size,
                          neighborhood_function=method)
        hc.initialize()
        hc.search(n_gen, False)
        print("Counter: "+str(counter)+" Seed:"+str(seed)+ " Time elapsed:"+str(datetime.now() - algo_start_time))
        hc.best_solution.print_()
        print("Training fitness of the best solution: %.2f" % hc.best_solution.fitness)

        run_results.append(hc.best_solution.fitness)

    time_elapsed = datetime.now() - start_time
    print("Run results: "+str(np.mean(run_results)))
    print(run_results)
    result_string = ";".join(["Run:"+str(counter),str(n_gen),str(neighborhood_size),str(np.mean(run_results)),str(time_elapsed)])
    with open("joris_tests.csv", "a") as myfile:
        myfile.write(result_string + "\n")
    counter+=1


# analyze data-frame
print("Results:\n"+str(run_results))
print("Descriptive statistics:\n"+str(results.describe()))

