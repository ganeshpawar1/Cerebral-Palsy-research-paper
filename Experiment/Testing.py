import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Append paths for custom modules
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\Model")
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\helper")

from LSTM import LSTMAutoencoder
from utils import reshape_patient_data, plot_reconstruction_error, plot_reconstruction_graph, denormalize  # Optional plotting util

class TestLSTM:
    def __init__(self, model_path, data_path, save_path='', idx_range=5, model='LSTM', units=4):
        self.model_path = model_path
        self.data_path = data_path
        self.save_path = save_path
        self.idx_range = idx_range
        self.folder_name = save_path + rf'\{model}\model_{units}_units'

        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = np.array(data.iloc[:, 1:])  # skip ID/index column
        X, hip_scaler, knee_scaler = reshape_patient_data(X)
        return X, hip_scaler, knee_scaler

    def evaluate(self):
        print("Loading model...")
        model = load_model(self.model_path, compile=False)  # compile=False if only for inference
        X, hip_scaler, knee_scaler = self.load_data()

        print("Predicting...")
        X_pred = model.predict(X)

        # Reconstruction error
        reconstruction_error = np.mean(np.square(X - X_pred), axis=(1, 2))  # per sample
        print("Reconstruction error shape:", reconstruction_error.shape)

        # Optionally plot reconstruction error histogram
        plot_reconstruction_error(X, X_pred, save_path=self.save_path)  # if implemented
        print("Plotting reconstruction(scaled) graph...")
        plot_reconstruction_graph(X, X_pred, idx_range=self.idx_range, y_label='Angle', save_path=self.folder_name)  # Optional plotting util

        print("Plotting reconstruction(original scaled) graph...")
        X_original = denormalize(X, hip_scaler, knee_scaler)
        X_pred_original = denormalize(X_pred, hip_scaler, knee_scaler)
        plot_reconstruction_graph(X_original, X_pred_original, idx_range=self.idx_range, y_label='Angle', save_path=self.folder_name)  # Optional plotting util

        return reconstruction_error, X, X_pred

if __name__ == '__main__':
    model_path = r"C:\Users\Admin\Desktop\Research paper\Training\Model\lstm_autoencoder_model_4_units_v1.keras"
    test_data_path = r"C:\Users\Admin\Desktop\Research paper\Data\test.csv"
    save_path = r"C:\Users\Admin\Desktop\Research paper\Testing\plots"

    tester = TestLSTM(model_path, test_data_path, save_path, idx_range=30)
    errors, X_true, X_reconstructed = tester.evaluate()
