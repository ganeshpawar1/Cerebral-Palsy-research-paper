from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
import sys
import os
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\Model")
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\helper")  
import numpy as np
import pandas as pd
from LSTM import LSTMAutoencoder  # Import your model class
from utils import plot_loss, reshape_patient_data  # Import your plotting function
from sklearn.preprocessing import MinMaxScaler


def build_model(hp):
    lstm_units = [hp.Int(f'units_{i}', min_value=16, max_value=128, step=8) for i in range(3)]
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    latent_dim = hp.Int('latent_dim', 8, 64, step=8)

    model = LSTMAutoencoder(
        input_shape=(51, 2),
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        l2_reg=1e-4,
        use_attention=True
    ).model
    model.compile(optimizer='adam', loss='mse')
    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=15,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='lstm_autoencoder_3_units'
)

# Load your data
data = pd.read_csv(r'C:\Users\Admin\Desktop\Research paper\Data\val.csv')  # Replace with your data path
patient_id = data['Patient ID']
X = data.drop(columns=['Patient ID'])
X_val = np.array(X)

print("X.shape:", X.shape)
print("X_val.shape:", X_val.shape)
# Normalize the data
scaler = MinMaxScaler() 
X_val = scaler.fit_transform(X_val)
# Reshape the data to match the input shape of the model
X_val = reshape_patient_data(X_val)
print("X_val.shape after reshaping:", X_val.shape)

tuner.search(X_val, X_val, epochs=30, validation_split=0.2)
best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print("best hyperparameters:",best_hyperparameters)

# Save the best model
best_model.save(r'C:\Users\Admin\Desktop\Research paper\HP_results\best_lstm_attention_autoencoder_3_units.h5') 
best_model.save(r'C:\Users\Admin\Desktop\Research paper\HP_results\best_lstm_attention_autoencoder_3_units.keras')  # Save in Keras format
# Save the best hyperparameters
#best_hyperparameters.save(r'C:\Users\Admin\Desktop\Research paper\HP_results\best_hyperparameters(LSTM+attention).json')
# Save the tuner results
tuner.results_summary()
# plot the loss curves

history = best_model.fit(X_val, X_val, epochs=30, validation_split=0.2, verbose=1)
plot_loss(history, label_1='Training Loss', label_2='Validation Loss')
# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(r'C:\Users\Admin\Desktop\Research paper\HP_results\val_training_history(LSTM+attention)_3_units.csv', index=False)