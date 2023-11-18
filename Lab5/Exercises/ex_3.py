from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

dataset = pd.read_csv("diabetes.csv")

(train_data, test_data, train_labels, test_labels) = train_test_split(random_state=231056, test_size=0.7)

# scaling the data
scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

def run():
    print(f'hidden_1 neurons = 2, hidden 2 neurons = 2')

def neural_network():
    # creating an classifier from the model:
    mlp = MLPClassifier(hidden_layer_sizes=(6, 2), activation='relu')

    # let's fit the training data to our model
    mlp.fit(train_data, train_labels)

    predictions_train = mlp.predict(train_data)
    print(accuracy_score(predictions_train, train_labels))
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels))

    confusion_matrix(predictions_train, train_labels)
    print(classification_report(predictions_test, test_labels))

