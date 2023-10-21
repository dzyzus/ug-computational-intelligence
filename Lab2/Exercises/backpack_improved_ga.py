import pygad
import time
from Lab2.Exercises.Models import backpack

S = backpack.create_backpack()
number_of_executions = 10
backpack_max_weight = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#kiedy stop
stop_criteria = "reach_1600"

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    current_weight = 0
    current_value = 0

    for item in range(len(S)):
        if solution[item] == 1:
            current_weight += S[item].weight
            current_value += S[item].value

    if current_weight > backpack_max_weight:
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

#uruchomienie algorytmu
def run_algorithm():
    global_time = 0
    for iteration in range(number_of_executions):
        global ga_instance
        global solution, solution_fitness, solution_idx

        #inicjalizacja ga_instance po każdej iteracji
        #w innym przypadku jak chcemy odczytać ile generacji nam to zajęło, zwróci pierwszą poprawnie
        #reszta będzie inkrementowana o 1
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
                               stop_criteria=stop_criteria,
                               mutation_percent_genes=mutation_percent_genes)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        #pomiar czasu dla instancji
        start = time.time()
        ga_instance.run()
        end = time.time()
        global_time += end-start

        print(f"\nAlgorithm execution time: {end-start}")
        print(f"{stop_criteria} value takes {ga_instance.generations_completed} generations")
    print(f"Average time of 10 executions: {global_time/number_of_executions}")

    #jeśli chcemy wypisywać podsumowanie i rysować za każdym razem wykres przenieś to do pętli for
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