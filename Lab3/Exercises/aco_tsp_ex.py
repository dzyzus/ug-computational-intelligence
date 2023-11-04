import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (28, 30),
    (41, 52),
    (89, 102),
    (105, 72),
    (81, 23),
    (85, 84)
)

"""
COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
)
"""

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


# Alfa - feromony mają wieksze znaczenie
# Beta - krawędź ma większe znaczenie

def run_aco():
    plot_nodes()

    colony = AntColony(COORDS, ant_count=150, alpha=1.0, beta=0.1,
                       pheromone_evaporation_rate=0.05, pheromone_constant=200, iterations=30)

    optimal_nodes = colony.get_path()

    for i in range(len(optimal_nodes) - 1):
        plt.plot(
            (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
            (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
        )


    plt.show()