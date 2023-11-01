import numpy
import math
from matplotlib import pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.5}

x_min = numpy.zeros(6)
x_max = numpy.ones(6)
my_bounds = (x_min, x_max)

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# Endurance func
def endurance(params):
    x, y, z, u, v, w = params
    return -(math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w))

# Compute whole swarm
def f(x):
    results = []
    for particle in x:
        result = endurance(particle)
        results.append(result)
    return numpy.array(results)

def run_pso():
    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=1000)
    # Obtain cost history from optimizer instance
    draw_plot()
    animation()

def draw_plot():
    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history
    # Plot!
    plot_cost_history(cost_history)
    plt.show()

def animation():
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.5}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    optimizer.optimize(fx.sphere, iters=50) 
    # tworzenie animacji 
    m = Mesher(func=fx.sphere) 
    animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
    animation.save("pso_sphere.gif", writer='imagemagick', fps=10)