"""
W tym zadaniu zrobimy parę operacji na bazie danych i stworzymy wykres. Możesz
wykorzystać paczkę pandas do obsługi baz danych i matplotlib do robienia wykresu1
.
a) Załaduj plik miasta.csv do programu w Pythonie i zapisz go pod nazwą miasta. Wyświetl
tabelę z danymi: miasta. Sprawdź jak wyglądają czyste wartości tej tabeli tzn.
miasta.values
b) Dodaj za pomocą odpowiedniej instrukcji wiersz do tabeli z ludnością w 2010 roku:
2010,460,555,405
c) Stwórz wykres dla ludności Gdańska (skorzystaj z paczki matplotlib). Dodaj odpowiednie
oznaczenia osi, tytuły. Wykres ma być liniowy z punktami i w kolorze czerwonym.
Powinno wyjść coś następującego:
d) Stwórz dodatkowo wykres, który będzie zestawiał zmiany ludności wszystkich miasta na
jednym wykresie w różnych kolorach. Dodaj legendę.
"""
import os
import pandas as pd
import matplotlib.pyplot as mplib
import numpy as np

def get_path(file):
    currentDirectory = os.getcwd()
    return f'{currentDirectory}\Exercises\Source\{file}'

def read_csv_file(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f'An error occurred: {str(e)}')

def add_row(path, data):
    try:
        file = read_csv_file(path)
        newRow = [data[0], data[1], data[2], data[3]]
        file = file._append(pd.Series(newRow, index=file.columns), ignore_index=True)
        file.to_csv(path, index=False)
    except Exception as e:
        print(f'An error occurred: {str(e)}')

def draw_plot_of_selected_city(file, city):
    dataFrame = pd.DataFrame(file)
    mplib.figure(figsize=(10, 6))
    mplib.plot(dataFrame['Rok'], dataFrame[f'{city}'], marker='o', color='red', label='Gdansk')
    mplib.title('Population of Gdansk')
    mplib.xlabel('Year')
    mplib.ylabel('Population')
    mplib.grid(True)
    mplib.legend()
    mplib.show()


def draw_plot_of_all_cities(file):
    dataFrame = pd.DataFrame(file)
    mplib.figure(figsize=(10, 6))

    for column in file:
        if (column == "Rok"):
            continue
        mplib.plot(dataFrame['Rok'], dataFrame[column], marker='o', color=random_color_generator(), label=column)

    mplib.title('Population of cities')
    mplib.xlabel('Year')
    mplib.ylabel('Population')
    mplib.grid(True)
    mplib.legend()
    mplib.show()

def random_color_generator():
    return np.random.rand(3)
