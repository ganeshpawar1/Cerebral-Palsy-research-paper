import sys
import os
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\Model")
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\helper") 

from LSTM import LSTMAutoencoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import plot_loss, reshape_patient_data

# LSTM Autoencoder Training Class
class TrainLSTM:
    def __init__(self, data_path, epochs=30, batch_size=32,model_path=None,save_path='',plot_save_path=''):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path if model_path else 'lstm_autoencoder_model.h5'
        self.save_path = save_path
        self.plot_save_path = plot_save_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = np.array(data.iloc[:, 1:])  # Assuming the first column is an index or ID
        print("X shape and dim",X.shape,X.ndim)
        X,hip_scaler,knee_scaler = reshape_patient_data(X)
        print("after reshape X shape and dim",X.shape,X.ndim)
        return X, hip_scaler, knee_scaler

    def train_model(self):
        X, hip_scaler, knee_scaler = self.load_data()
        X_train, X_val = train_test_split(X, test_size=0.3, random_state=42)
        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_train dim:", X_train.ndim)
        print("X_val dim:", X_val.ndim)
        print("X_train nan values",np.any(np.isnan(X_train)), np.any(np.isinf(X_train)))
        print("X_val nan values",np.any(np.isnan(X_val)), np.any(np.isinf(X_val)))

        model = LSTMAutoencoder(input_shape=X_train.shape[1:])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, X_val))
        print('history:',history)
        plot_loss(history,save_path=self.plot_save_path)
        model.save(self.save_path)
        print("model saved at:", self.save_path)

if __name__ == '__main__':
    data_path = r'C:\Users\Admin\Desktop\Research paper\Data\train.csv' 
    model_path = r'C:\Users\Admin\Desktop\Research paper\HP_results\LSTM\best_lstm_attention_autoencoder_4_units_v1.h5'
    save_path = r'C:\Users\Admin\Desktop\Research paper\Training\Model\lstm_autoencoder_model_4_units_v1.keras'
    plot_save_path = r'C:\Users\Admin\Desktop\Research paper\Training\loss plots\training_loss_plot_4_units_v1.png'
    trainer = TrainLSTM(data_path, epochs=50, batch_size=8, model_path=model_path, save_path=save_path, plot_save_path=plot_save_path)
    trainer.train_model()