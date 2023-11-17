from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
from sklearn import tree
import graphviz 

# Install windows package from: https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# Install python graphviz package
# Add C:\Program Files (x86)\Graphviz2.38\bin to User path
# Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path

iris = datasets.load_iris()
print(iris)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=231056)

# Init decision tree
clf = DecisionTreeClassifier()

def run():
    clf.fit(X_train, y_train)
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render(filename='iris_decision_tree', format='png', cleanup=True)
    tree.plot_tree(clf)
    constructor_evaluation()

def constructor_evaluation():
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    Error(y_pred)

def Error(y_pred):
    error_matrix = confusion_matrix(y_test, y_pred)
    print('Error Matrix:')
    print(error_matrix)
