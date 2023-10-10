from Exercises import PrimeNumber
from Exercises import MathematicalStatistics
from Exercises import PlotLib

"""
prime(3)
Wynik: True
prime(4)
Wynik: False
prime(49)
Wynik: False
"""

print("Exercise - 1 a)\n")
PrimeNumber.prime(3)
PrimeNumber.prime(4)
PrimeNumber.prime(49)

"""
Przykład działania:
select_primes([3, 6, 11, 25, 19])
Wynik: [3, 11, 19]
"""

print("\nExercise - 1 b)\n")
PrimeNumber.select_primes([3, 6, 11, 25, 19])

v1 = [3, 8, 9, 10, 12]
v2 = [8, 7, 7, 5, 6]

print("\nExercise - 2 a)\n")
MathematicalStatistics.vector_sum(v1, v2)
MathematicalStatistics.vector_product(v1, v2)

print("\nExercise - 2 b)\n")
MathematicalStatistics.dot_product(v1, v2)

print("\nExercise - 2 c)\n")
MathematicalStatistics.euclidean_distance(v1, v2)

print("\nExercise - 2 d)\n")
randomVector = MathematicalStatistics.random_vector(numberOfRandomValues=50)
print(f'Generated vector: {randomVector}')

print("\nExercise - 2 e)\n")
MathematicalStatistics.vector_details(randomVector)

print("\nExercise - 2 f)\n")
MathematicalStatistics.normalize_vector(randomVector)

print("\nExercise - 2 g)\n")
MathematicalStatistics.standardization_vector_numpy(randomVector)
MathematicalStatistics.standardization_vector_manual(randomVector)

print("\nExercise - 3 a)\n")

path = PlotLib.get_path("miasta.csv")
cities = PlotLib.read_csv_file(path)
print(cities)
print(f'\nOnly values: \n{cities.values}')

print("\nExercise - 3 b)\n")
PlotLib.add_row(path, ["2010", "460", "555", "405"])
cities = PlotLib.read_csv_file(path)
print(cities)

