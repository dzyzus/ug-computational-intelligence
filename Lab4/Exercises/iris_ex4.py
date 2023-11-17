from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)

# Prediction and acc of decision tree
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# k-NN cases
k_values = [3, 5, 11]

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    knn_classifier.fit(X_train, y_train)
    
    # Prediction and acc on KNN
    y_pred_knn = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    
    print(f'k-NN (k={k}) Accuracy: {accuracy_knn * 100:.2f}%')
    print(f'k-NN (k={k}) Confusion Matrix:')
    print(conf_matrix_knn)
    print('-' * 50)

# Naive Bayes
nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

# Prediction and acc of NB
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

def run():
    print(f'NB Accuracy: {accuracy_nb * 100:.2f}%')
    print(f'NB Confusion Matrix:')
    print(conf_matrix_nb)

    print('\nComparison of classifier accuracy:')
    print(f'DT acc: {accuracy_dt * 100:.2f}%')
    for k in k_values:
        print(f'k-NN (k={k}) acc: {accuracy_knn * 100:.2f}%')
    print(f'NB acc: {accuracy_nb * 100:.2f}%')