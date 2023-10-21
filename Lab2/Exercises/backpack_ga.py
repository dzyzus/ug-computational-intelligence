import random
import pygad
import numpy
from Lab2.Exercises.Models import backpack

S = backpack.create_backpack()

backpack_max_weight = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    current_weight = 0
    current_value = 0

    for item in range(len(S)):
        if solution[item] == 1:
            current_weight += S[item].weight
            current_value += S[item].value

    if current_weight >= backpack_max_weight:
        return 0

    return current_value

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 50
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 10
num_generations = 20
keep_parents = 10

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

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
    summary = ga_instance.best_solution()
    print(f"Parameters of the best solution generation : {summary}\n")
    print_items(summary[0])
    print(f"Value of items: {summary[1]}")

def draw_plot():
    #wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
    ga_instance.plot_fitness()

def print_items(solution):
    selected_items = []
    weight = 0
    for item in range(len(S)):
        if solution[item] == 1.:
            selected_items.append(S[item])
            weight += S[item].weight
    for item in selected_items:
        print(f"Nazwa: {item.name}, Waga: {item.weight}, Cena: {item.value}")

    print(f"\nWeight of items: {weight}")

