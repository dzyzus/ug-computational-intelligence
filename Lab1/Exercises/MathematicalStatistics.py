"""
a) Wczytaj dwa wektory z liczbami [3, 8, 9, 10, 12] oraz [8, 7, 7, 5, 6] (jako pythonowe listy).
Następnie zwróć sumę tych wektorów oraz iloczyn (po współrzędnych) tych wektorów.
b) Dla powyższych wektorów podaj iloczyn skalarny.
c) Dla powyższych wektorów podaj ich długości euklidesowe (długość wektora jako strzałki w
przestrzeni).
d) Stwórz wektor 50 losowych liczb z zakresu od 1 do 100.
e) Dla wektora z punktu (d) policz średnią z wszystkich jego liczb, min, max oraz odchylenie
standardowe.
f) Dokonaj normalizacji wektora z podpunktu (d) (ściskamy wszystkie liczby do przedziału
[0,1]) za pomocą poniższego wzoru (xi to liczba w starym wektorze na pozycji i, a zi to liczba
w nowym wektorze na pozycji i)
W oryginalnym wektorze jakie było max? Na której pozycji stało? Jaka liczba stoi na tej
pozycji w nowym wektorze?
g) Dokonaj standaryzacji wektora z podpunktu (d). Wzór na standaryzację wykorzystuje
średnią i odchylenie standardowe:
𝑧𝑖 =
𝑥𝑖 − 𝑚𝑒𝑎𝑛(𝑥)
σ(x)
Jaką średnią i odchylenie standardowe ma nowy wektor z?
"""
import math
import random

def vector_sum(v1, v2):
    print(f'V1 sum: {sum(v1)}\nV2 sum: {sum(v2)}\nSum V1+V2: {sum(v1) + sum(v2)}')

def vector_product(v1, v2):
    iteration = 1
    zip_object = zip(v1, v2)
    for val1, val2 in zip_object:
        print(f'val1_{iteration} - {val1} from v1 * val2 - {val2} from v2_{iteration} = {val1 * val2}')
        iteration += 1

def dot_product(v1, v2):
    zip_object = zip(v1, v2)
    value = 0
    for val1, val2 in zip_object:
        value += val1 * val2
    print(f'Dot product of {v1} and {v2} is equal {value}')

def euclidean_distance(v1, v2):
    iteration = 1
    zip_object = zip(v1, v2)
    value = 0
    for val1, val2 in zip_object:
        value += (val1 - val2)**2
    print("Source: https://en.wikipedia.org/wiki/Euclidean_distance")
    print(f'Euclidean distance is equal: {math.sqrt(value)}')

def random_vector(numberOfRandomValues):
    vector = []
    for i in range(0, numberOfRandomValues):
        randomNumber = random.randint(1, 100)
        vector.append(randomNumber)
    return vector

def vector_details(v1):
    print(f'Vector: {v1}')
    print(f'Average value: {average(v1)}')
    print(f'The min value: {min(v1)}')
    print(f'The max value: {max(v1)}')
    print(f'The standard deviation: {math.sqrt(average(v1))}')

def average(v1):
    return sum(v1)/len(v1)

def normalize_vector(v1):
    vectorLength = 0
    for value in v1:
        vectorLength += value**2
    vectorLength = math.sqrt(vectorLength)
    normalized_vector = []
    for var in v1:
        normalized_vector.append(var/vectorLength)
    print(f'V1: {v1}\nNormalized vector: {normalized_vector}')

    """
    todo index of max value vs value from normalized vecotr
    """
