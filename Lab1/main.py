from Exercises import PrimeNumber
from Exercises import MathematicalStatistics

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
print(f'Generated vector: {MathematicalStatistics.random_vector(numberOfRandomValues=50)}')

print("\nExercise - 2 e)\n")
MathematicalStatistics.vector_details(MathematicalStatistics.random_vector(numberOfRandomValues=50))
