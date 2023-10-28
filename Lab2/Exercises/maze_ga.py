import pygad
import numpy as np
import time

maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

start = (1, 1)
end = (10, 10)
max_moves = 30
max_moves_to_point = 0
# kiedy stop
stop_criteria = "reach_1000"

# definiujemy parametry chromosomu
# 0 - góra, 1 - dół, 2 - prawo
gene_space = [0, 1, 2]


def calculate_distance(x, y):
    x1, y1 = end
    diff_x = x1 - x
    diff_y = y1 - y

    return int(round(np.linalg.norm([diff_x, diff_y])))


# definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    global max_moves_to_point
    x, y = start
    moves = 0
    total_distance = 0

    for move in solution:
        if move == 0:
            y -= 1
        elif move == 1:
            y += 1
        elif move == 2:
            x += 1

        moves += 1

        if (x, y) == end:
            max_moves_to_point = moves
            return 1000

        if x < 0 or x >= len(maze[0]) or y < 0 or y >= len(maze) or maze[y][x] == 0:
            total_distance -= 10
            if move == 0:
                y += 1
            elif move == 1:
                y -= 1
            elif move == 2:
                x -= 1

        total_distance += calculate_distance(x, y)

        if moves >= max_moves:
            return total_distance


fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 500
num_genes = max_moves

# ile wylaniamy rodzicow do "rozmanazania"
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 30
num_generations = 1500
keep_parents = 5

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "rank"

# w ilu punktach robic krzyzowanie?
crossover_type = "two_points"

# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10


# inicjacja populacji

def initialize_population():
    population = []
    for pop in range(max_moves):
        solution = np.random.choice(gene_space, size=max_moves)
        population.append(solution)
    return population


def run_algorithm(number_of_executions: int):
    global_time = 0
    for iteration in range(number_of_executions):
        global ga_instance
        global solution, solution_fitness, solution_idx

        # inicjalizacja ga_instance po każdej iteracji
        # w innym przypadku jak chcemy odczytać ile generacji nam to zajęło, zwróci pierwszą poprawnie
        # reszta będzie inkrementowana o 1
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
                               mutation_percent_genes=mutation_percent_genes,
                               initial_population=initialize_population(),
                               stop_criteria=stop_criteria)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        # pomiar czasu dla instancji
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        global_time += end_time - start_time

        print(f"\nAlgorithm execution time: {end_time - start_time}")
        if (ga_instance.best_solution()[1] == 1000):
            print(f"{stop_criteria} value takes {ga_instance.generations_completed} generations")
        else:
            print(f"The {stop_criteria} didn't reached even in {ga_instance.generations_completed} generations")


    print(f"Average time of 10 executions: {global_time / number_of_executions}")

    # jeśli chcemy wypisywać podsumowanie i rysować za każdym razem wykres przenieś to do pętli for
    summary()
    draw_plot()


# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
def summary():
    summary = ga_instance.best_solution()
    print(f"Parameters of the best solution generation : {summary}\n")
    print(f"Fitness: {summary[1]}")
    print_moves()


def draw_plot():
    # wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
    ga_instance.plot_fitness()


# 0 - góra, 1 - dół, 2 - prawo
def print_moves():
    global max_moves_to_point
    moveList = []
    summary = ga_instance.best_solution()

    if max_moves_to_point == 0:
        max_moves_to_point = max_moves

    print(f"It takes {max_moves_to_point} moves to reach {end}")

    for move in ga_instance.best_solution()[0][:max_moves_to_point]:
        if move == 0:
            moveList.append("UP")
        if move == 1:
            moveList.append("DOWN")
        if move == 2:
            moveList.append("RIGHT")
    print(moveList)
