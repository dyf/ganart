import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

def MorphModel(input_shape):
    return Sequential([
        LSTM(128, activation='relu', input_shape=(input_shape[0], input_shape[1]), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])
    