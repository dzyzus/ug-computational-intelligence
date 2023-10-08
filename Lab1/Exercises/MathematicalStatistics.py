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
