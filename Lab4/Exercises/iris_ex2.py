import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=231056)

def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return("Setosa")
    elif pw > 2:
        return("Virginica")
    else:
        return("Versicolor")

def check_all():
    good_predictions = 0
    len = test_set.shape[0]
    for i in range(len):
        sl, sw, pl, pw, actual_class = test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3], test_set[i, 4]
        if classify_iris(sl, sw, pl, pw) == actual_class:
            good_predictions = good_predictions + 1
    print(good_predictions)
    print(good_predictions/len*100, "%")