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

PrimeNumber.prime(3)
PrimeNumber.prime(4)
PrimeNumber.prime(49)

"""
Przykład działania:
select_primes([3, 6, 11, 25, 19])
Wynik: [3, 11, 19]
"""

PrimeNumber.select_primes([3, 6, 11, 25, 19])

v1 = [3, 8, 9, 10, 12]
v2 = [8, 7, 7, 5, 6]

MathematicalStatistics.vector_sum(v1, v2)
MathematicalStatistics.vector_product(v1, v2)
