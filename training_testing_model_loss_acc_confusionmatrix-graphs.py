import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import NUM_EPOCH, get_model
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_actions, get_sequences_and_labels
from constants import MAX_LENGTH_FRAMES, MODEL_NAME

def training_model(data_path, model_path):
    actions = get_actions(data_path)
    # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES,padding='post', truncating='post', dtype='float32')

    # Get sequence data and labels
    X = np.array(sequences)                 # input features (Numpy array)
    y = to_categorical(labels).astype(int)  # labels
    print(y)

    # Split data into training and test sets
    # test_size=0.1=10% for testing and the rest (90%) for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Define the model and compile it
    model = get_model(len(actions))

    # Train the model and save history
    history = model.fit(X_train, y_train, epochs=NUM_EPOCH, validation_data=(X_test, y_test))

    # Get training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create loss metrics graphs
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Create accuracy metrics graphs
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Calculate predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=actions, columns=actions)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Calculate precision, recall, and F1-score
    report = classification_report(y_test, y_pred, target_names=actions)
    print("Classification Report:")
    print(report)

    # Show model summary
    model.summary()

    # Save the model
    model.save(model_path)

if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root, "data")
    save_path = os.path.join(root, "models")
    model_path = os.path.join(save_path, MODEL_NAME)
    
    training_model(data_path, model_path)
