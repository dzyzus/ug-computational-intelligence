from Helpers.process_data import ProcessData
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

epochs = 1000
batch_size = 20
model_filename = 'thyroid-classifier-model.h5'

def create_keras_model(input_dim, num_classes=3):
    model = Sequential()

    # First hidden layer
    model.add(Dense(48, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))

    # Dropout 25 % of neurons
    model.add(Dropout(0.25))

    # Second hidden layer
    model.add(Dense(48, kernel_initializer='uniform', activation='relu'))

    # Dropout 25 % of neurons
    model.add(Dropout(0.25))

    # Third hidden layer
    model.add(Dense(48, kernel_initializer='uniform', activation='relu'))

    # Dropout 25 % of neurons
    model.add(Dropout(0.25))

    # Output layer with softmax activation for multi-class classification
    model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

    # Create or load model
    input_dim = X_train_scaled.shape[1]
    num_classes = 3  # 3 classes

    # Load model and train on it
    if os.path.exists(model_filename):
        model = data.load_saved_model(model_filename)
    else:
        model = create_keras_model(input_dim, num_classes)

    # Create classifier
    classifier = KerasClassifier(build_fn=lambda: model, epochs=epochs, batch_size=batch_size, verbose=1)

    # Train the model with validation data
    history = classifier.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test))

    # Save the model
    data.save_model(classifier.model, model_filename)

    # Prediction on Test data
    testLabelPredicted = classifier.predict(X_test_scaled)

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

    # Calculate accuracy
    accuracy = accuracy_score(y_test, testLabelPredicted)
    print(f"Accuracy: {accuracy * 100:.2f}%")


    # Draw plots
    draw_plot(history)
    plot_confusion_matrix(conf_matrix, target_names=['Healthy', 'Hypothyroid', 'Hyperthyroid'])

    # Prompt user for self-check
    check_myself = input("Do you want to check yourself? (Y/N): ").upper()

    if check_myself == "Y":
        data.check_thyroid_status(model, scaler)
    else:
        print("Okay, no problem. If you change your mind, feel free to check later.")

def draw_plot(history):
    # Plotting accuracy and loss for both training and validation data
    plt.figure(figsize=(12, 5))

    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save and display the plots
    plt.savefig("model_performance_plot.png")
    plt.show()

def plot_confusion_matrix(conf_matrix, target_names):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ConfusionMatrixDisplay(conf_matrix, display_labels=target_names).plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_plot_neural_network.png")
    plt.show()