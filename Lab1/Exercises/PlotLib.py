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
import matplotlib as mp

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
        row = {'Rok': data[0], 'Gdansk': data[1], 'Poznan': data[2], 'Szczecin': data[3]}
        newRow = pd.DataFrame([row])
        file = pd.concat([file, newRow], ignore_index=True)
        file.to_csv(path, index=True)
        read_csv_file(path)
    except Exception as e:
        print(f'An error occurred: {str(e)}')

def draw_plot():
