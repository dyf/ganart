import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional, Input, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

def MorphModel(input_shape):
    return Sequential([
        GRU(512, activation='relu', input_shape=(input_shape[0], input_shape[1]), return_sequences=True),
        GRU(256, activation='sigmoid', return_sequences=False),
        Dropout(0.2),
        RepeatVector(input_shape[0]),
        GRU(256, activation='relu', return_sequences=True),
        GRU(512, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])
    