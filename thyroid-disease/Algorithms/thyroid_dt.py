from Helpers.process_data import ProcessData
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    # Initialize ProcessData class
    data = ProcessData()

    # Load Datasets
    df = data.load_data()

    # Process data
    df = data.process_data(df)

    # Split data
    X_train, X_test, y_train, y_test = data.split_to_train_and_test(df)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)

    # Use Decision Tree for predictions
    testLabelPredicted = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, testLabelPredicted)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, testLabelPredicted)

    # Extract values from the confusion matrix
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tp = conf_matrix[1, 1]

    print("----------------------------------------------------------")
    print("Confusion matrix")
    print("----------------------------------------------------------")
    print(f'True Positives: {tp}')
    print(f'True Negatives: {tn}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')

    # Additional Predictions
    hypothyroid_predicted = (testLabelPredicted == 1)
    hyperthyroid_predicted = (testLabelPredicted == 2)
    healthy_predicted = (testLabelPredicted == 0)

    print("----------------------------------------------------------")
    print("Predicted")
    print("----------------------------------------------------------")
    print(f'Hypothyroid Predicted: {hypothyroid_predicted.sum()}')
    print(f'Hyperthyroid Predicted: {hyperthyroid_predicted.sum()}')
    print(f'Healthy Predicted: {healthy_predicted.sum()}')

    # Count actual cases
    hypothyroid_actual = (y_test.values == 1)
    hyperthyroid_actual = (y_test.values == 2)
    healthy_actual = (y_test.values == 0)

    print("----------------------------------------------------------")
    print("Actual")
    print("----------------------------------------------------------")
    print(f'Actual Hypothyroid Cases: {hypothyroid_actual.sum()}')
    print(f'Actual Hyperthyroid Cases: {hyperthyroid_actual.sum()}')
    print(f'Actual Healthy Cases: {healthy_actual.sum()}')

    # Classification Report
    class_report = classification_report(y_test, testLabelPredicted, target_names=['Healthy', 'Hypothyroid', 'Hyperthyroid'])
    print("----------------------------------------------------------")
    print('Classification Report:')
    print(class_report)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues", xticklabels=['Healthy', 'Hypothyroid', 'Hyperthyroid'],
                yticklabels=['Healthy', 'Hypothyroid', 'Hyperthyroid'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix_plot_dt.png")
    plt.show()
    

    data.save_decision_tree_visualization(model, target_names=['Healthy', 'Hypothyroid', 'Hyperthyroid'])

    # Prompt user for self-check
    check_myself = input("Do you want to check yourself? (Y/N): ").upper()

    if check_myself == "Y":
        data.check_thyroid_status(model, scaler)
    else:
        print("Okay, no problem. If you change your mind, feel free to check later.")
