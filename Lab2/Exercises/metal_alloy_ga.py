import pygad
import math
import random
from Exercises.Models import metals

S = metals.create_set()

# definiujemy parametry chromosomu
# geny to liczby: 0 lub 1
gene_space = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


# definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution
    max_value = max(solution)
    min_valu = min(solution)
    if max_value > 1.0 or min_valu < 0.0:
        return 0
    print(solution)
    return endurance(*solution)


fitness_function = fitness_func


def initialize_population(sol_per_pop, gene_space):
    population = []
    for pop in range(sol_per_pop):
        individual = [random.uniform(min_value, max_value) for (min_value, max_value) in gene_space]
        population.append(individual)
    return population


# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 100
num_genes = len(S)

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 30
num_generations = 50
keep_parents = 10

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "roulette"

# w ilu punktach robic krzyzowanie?
crossover_type = "single_point"

# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 16.67

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       initial_population=initialize_population(sol_per_pop, gene_space))

solution, solution_fitness, solution_idx = ga_instance.best_solution()


# uruchomienie algorytmu
def run_algorithm():
    ga_instance.run()
    summary()
    draw_plot()


# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
def summary():
    summary = ga_instance.best_solution()
    print(f"Parameters of the best solution generation : {summary[0]}\n")
    print(f"Max durability: {summary[1]}")

def draw_plot():
    # wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
    ga_instance.plot_fitness()