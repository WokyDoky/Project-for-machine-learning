import warnings
warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Define constants
DATASET_PATH = '../Project-for-machine-learning/archive/genres'
GENRES = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
SAMPLE_DURATION = 2
HOP_LENGTH = 256
N_FFT = 512
N_MELS = 64
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

# Function to load dataset
def load_dataset():
    dataset = []
    for genre, genre_number in GENRES.items():
        genre_path = os.path.join(DATASET_PATH, genre)
        for filename in os.listdir(genre_path):
            song_path = os.path.join(genre_path, filename)
            for index in range(14):
                y, sr = librosa.load(song_path, mono=True, duration=SAMPLE_DURATION, offset=index * SAMPLE_DURATION)
                ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)
                ps = librosa.power_to_db(ps**2)
                dataset.append((ps, genre_number))
    return dataset

# Function to prepare data for training
def prepare_data(dataset):
    random.shuffle(dataset)
    train = dataset[:10000]
    valid = dataset[10000:12000]
    test = dataset[12000:]

    X_train, Y_train = zip(*train)
    X_valid, Y_valid = zip(*valid)
    X_test, Y_test = zip(*test)

    X_train = np.array([x.reshape((N_MELS, -1, 1)) for x in X_train])
    X_valid = np.array([x.reshape((N_MELS, -1, 1)) for x in X_valid])
    X_test = np.array([x.reshape((N_MELS, -1, 1)) for x in X_test])

    Y_train = np.array(tf.keras.utils.to_categorical(Y_train, len(GENRES)))
    Y_valid = np.array(tf.keras.utils.to_categorical(Y_valid, len(GENRES)))
    Y_test = np.array(tf.keras.utils.to_categorical(Y_test, len(GENRES)))

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# Function to create model
def create_model():
    model = Sequential()
    # First Conv Block
    model.add(Conv2D(16, (5, 5), input_shape=(N_MELS, 173, 1), activation="relu", padding="valid", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Second Conv Block
    model.add(Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(len(GENRES), activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=['accuracy'])
    return model

# Function to plot training history
def plot_history(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title('Model Accuracy')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title('Model Loss')
    plt.show()

# Main function
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset loaded with {len(dataset)} samples.")

    print("Preparing data...")
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_data(dataset)
    print("Data preparation complete.")

    print("Creating model...")
    model = create_model()
    model.summary()

    print("Training model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_valid, Y_valid), callbacks=[early_stopping])

    print("Training complete. Plotting history...")
    plot_history(history)

    print("Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=2)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print(f'Training accuracy: {train_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()
