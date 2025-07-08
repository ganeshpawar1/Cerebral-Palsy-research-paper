# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import (
    LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Multiply, 
    Lambda, Dropout, BatchNormalization, TimeDistributed
)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Layer


#attention layer    
class AttentionLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        d_k = tf.cast(tf.shape(inputs)[-1], tf.float32)
        scores = tf.matmul(inputs, inputs, transpose_b=True)  # (batch, time, time)
        scaled_scores = scores / tf.math.sqrt(d_k)
        weights = tf.nn.softmax(scaled_scores, axis=-1)
        context = tf.matmul(weights, inputs)
        return context

class LSTMAutoencoder:
    def __init__(self, input_shape=(51, 2), latent_dim=16, lstm_units=[64, 32],dropout_rate=0.3, l2_reg=1e-4,use_attention=False):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.model, self.encoder = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = inputs

        # --- Encoder ---
        for i, units in enumerate(self.lstm_units):
            return_seq = True if i < len(self.lstm_units) - 1 else False
            x = LSTM(units, return_sequences=return_seq,
                     kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

        if self.use_attention:
            x = AttentionLayer()(x)

        latent = Dense(self.latent_dim, activation='relu',
                       kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        encoded = BatchNormalization()(latent)

        # --- Decoder ---
        x = RepeatVector(self.input_shape[0])(encoded)

        for i, units in enumerate(reversed(self.lstm_units)):
            x = LSTM(units, return_sequences=True,
                     kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        if self.use_attention:
            x = AttentionLayer()(x)

        outputs = TimeDistributed(Dense(self.input_shape[1]))(x)

        autoencoder = Model(inputs, outputs)
        encoder = Model(inputs, encoded)

        return autoencoder, encoder

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def latent_representation(self, X):
        return self.encoder.predict(X)


    def save(self, model_path='autoencoder_model.h5'):
        self.model.save(model_path)
