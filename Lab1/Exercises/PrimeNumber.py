"""
a) Stwórz funkcję prime(n), która będzie sprawdzała, czy podana liczba n jest liczbą
pierwszą. Jeśli tak, to zwróci True, w przeciwnym wypadku zwróci false.
Przetestuj czy funkcja działa dobrze np.
prime(3)
Wynik: True
prime(4)
Wynik: False
prime(49)
Wynik: False
b) Stwórz drugą funkcję select_primes(x), która dostanie listę x liczb naturalnych,
odfiltruje z niej wszystkie liczby pierwsze i zwróci listę liczb pierwszych.
W środku funkcji select_primes można użyć funkcji prime z podpunktu a.
Przykład działania:
select_primes([3, 6, 11, 25, 19])
Wynik: [3, 11, 19]
"""

def prime(n):
    if (n > 1):
        for value in range(2, n):
            if (n % value) == 0:
                print(f'{n} is not a prime number')
                return False
        else:
            print(f'{n} is a prime number')
            return True
    else:
        print(f'{n} will not be analyzed')
        return False

def select_primes(x):
    primeNumbers = []
    for val in x:
        if (prime(val)):
            primeNumbers.append(val)
    print(f'Prime numbers: {primeNumbers}')



