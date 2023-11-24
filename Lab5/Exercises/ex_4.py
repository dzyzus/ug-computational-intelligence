import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

dataset = pd.read_csv("diabetes.csv")

# Assuming df is your DataFrame loaded from the dataset
X = dataset.drop('class', axis=1)  # Features
y = dataset['class']  # Target variable

# Split the dataset into a training set and a test set
(train_data, test_data, train_labels, test_labels) = train_test_split(X, y, random_state=231056, test_size=0.7)

# Map 'string values' into int.
label_mapping = {'tested_negative': 0, 'tested_positive': 1}
train_labels_numeric = train_labels.map(label_mapping)
test_labels_numeric = test_labels.map(label_mapping)

# Is one complete pass through the entire training dataset
NB_EPOCHS = 500 
BATCH_SIZE = 16

def run():
    # Create our model
    model = Sequential()
    # AVAILABLE ACTIVATORS: 
    # sigmoid - 0/1 values
    # relu - max (0, x)
    # tanh (-1,1)
    # softmax - transform the results into a probability distribution
    # 1st hidden layer: 6 neurons, RELU
    model.add(Dense(6, input_dim=8, kernel_initializer='uniform', activation='relu'))
    # 2nd hidden layer: 3 neurons, RELU
    model.add(Dense(3, kernel_initializer='uniform', activation='relu'))
    # output layer: dim=1, activation sigmoid
    # must be sigmoid only 0/1 in that situation
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # AVAILABLE OPTIMIZERS
    # adam - combines algorithms adagrad and rmsprop
    # sgd - stochastic gradient descent
    # rmsprop - adaptive optimizer, adjusts learning rate for each parameter
    # adagrad - modifies learning rate based on previous gradients
    # adadelta - extension of adagrad that attempts to address its shortcomings
    # nadam - combination of adam and sgd with momentum
    # Compile the model
    # since we are predicting 0/1
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # checkpoint: store the best model
    # hdf - hierarchical data format version
    ckpt_model = 'weights.hdf5'
    checkpoint = ModelCheckpoint(ckpt_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # train the model, store the results for plotting
    history = model.fit(train_data,
                        train_labels_numeric.astype(int),
                        validation_data=(test_data, test_labels_numeric.astype(int)),
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=0)
    
    
    # Predictions and evaluation
    # > 0.5 -> classified as positive
    # ex. 8/8 total number of batches
    predictions_train = (model.predict(train_data) > 0.5).astype(int)
    print("Train Accuracy:", accuracy_score(predictions_train, train_labels_numeric))
    
    predictions_test = (model.predict(test_data) > 0.5).astype(int)
    print("Test Accuracy:", accuracy_score(predictions_test, test_labels_numeric))
    
    print(f"Confusion Matrix (Train): \n{confusion_matrix(predictions_train, train_labels_numeric)}")

    print_model(model)
    draw_plot(history)

def draw_plot(history):
    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])

    # save
    plt.savefig("relu_relu_sigmoid_sgd_500iteration")

    # Display the plots
    plt.show()

def print_model(model):
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

