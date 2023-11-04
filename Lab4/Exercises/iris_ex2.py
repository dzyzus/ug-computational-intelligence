import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=231056)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return("Setosa")
    elif pl <= 5:
        return("Virginica")
    else:
        return("Versicolor")

def check_all():
    good_predictions = 0
    len = test_set.shape[0]
    for i in range(len):
        if classify_iris(test_inputs.index[i], 0, test_classes.index[i], 0) == test_classes.index[i]:
            good_predictions = good_predictions + 1
    print(good_predictions)
    print(good_predictions/len*100, "%")