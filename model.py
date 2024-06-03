import keras.optimizers

from keras.models import Sequential
from keras.layers import LSTM, Dense
from constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 100

def get_model(output_lenght: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_lenght, activation='softmax'))
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #We add parameters to optimize the model
    #During training have a self-assessment, which is optimized with Adam, and the metric will be accuracy
    optimization = keras.optimizers.Adam(learning_rate= 0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])

    return model


    