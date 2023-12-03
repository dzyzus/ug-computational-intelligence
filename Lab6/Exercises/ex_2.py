import logging
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Logger
logging.basicConfig(filename='training.log', level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# Ustal ścieżkę do danych
dataset = 'dogs-cats-mini/'

# Only 25%
def load_partial_dataset(directory, percentage=0.25):
    logger.info("Wczytywanie obrazków.")
    X, y = list(), list()
    for subdir in listdir(directory):
        label = 0 if 'cat' in subdir else 1  # 0 - cat, 1 - dog
        files = listdir(directory + subdir)
        num_files_to_load = int(percentage * len(files))
        for file in files[:num_files_to_load]:
            filepath = directory + subdir + '/' + file
            image = load_img(filepath, target_size=(200, 200))
            image = img_to_array(image)
            X.append(image)
            y.append(label)
            logger.info(file)
    return X, y

# Load
logger.info("Wczytano dane.")
X, y = load_partial_dataset('dogs-cats-mini/train/')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja
logger.info("Dzielenie na zbiory train/test")
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# Zdefiniuj model CNN
logger.info("Definiowanie modelu")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout 50% loss
model.add(Dense(1, activation='sigmoid')) # 0/1

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
logger.info("Rozpoczynam trenowanie modelu.")
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint])
logger.info("Zakończono trenowanie modelu.")

# Plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
plt.savefig("lab6_ex2")

# Acc
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
logger.info(f"Dokładność na zbiorze testowym: {test_acc}")

# Confusion matrix
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
logger.info("Koniec.")