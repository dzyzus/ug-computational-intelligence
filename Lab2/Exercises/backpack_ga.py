from Exercises.Models import backpack
import random
import pygad
import numpy

S = [
    backpack.item('zegar', 100, 7),
    backpack.item('obraz-pejzaz', 300, 7),
    backpack.item('obraz-portret', 200, 6),
    backpack.item('radio', 40, 2),
    backpack.item('laptop', 500, 5),
    backpack.item('lampka nocna', 70, 6),
    backpack.item('srebrne sztucce', 100, 1),
    backpack.item('porcelana', 250, 3),
    backpack.item('figura z brazu', 300, 10),
    backpack.item('skorzana torebka', 280, 3),
    backpack.item('odkurzacz', 300, 15)]

backpack_max_weight = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    current_weight_s1 = 0
    current_weight_s2 = 0
    s1 = 0
    s2 = 0
    for item in S:
        if (current_weight_s1 <= backpack_max_weight):
            current_item = random.choice(S)
            current_weight_s1 += current_item.weight
            s1 += current_item.value
    for item in S:
         if (current_weight_s2 <= backpack_max_weight):
            current_item = random.choice(S)
            current_weight_s2 += current_item.weight
            s2 += current_item.value

    fitness = -numpy.abs(s1-s2)
    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty

ga_instance = pygad.GA(gene_space=gene_space,
                    num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_function,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_percent_genes=mutation_percent_genes)

solution, solution_fitness, solution_idx = ga_instance.best_solution()


#uruchomienie algorytmu
def run_algorithm():
    ga_instance.run()
    summary()
    draw_plot()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
def summary():
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

def draw_plot():
    #tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
    prediction = numpy.sum(S*solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    #wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
    ga_instance.plot_fitness()