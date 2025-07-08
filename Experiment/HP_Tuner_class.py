import os
import sys
import numpy as np
import pandas as pd
from kerastuner.tuners import BayesianOptimization,RandomSearch,GridSearch
from sklearn.preprocessing import MinMaxScaler
import yaml  # Make sure this is imported at the top
from tensorflow.keras.callbacks import EarlyStopping


# Add project-specific pathspip install tensorflow==2.15 keras==2.15 keras-tuner==1.4.5

sys.path.append(r"C:\Users\Admin\Desktop\Research paper\Model")
sys.path.append(r"C:\Users\Admin\Desktop\Research paper\helper")

from LSTM import LSTMAutoencoder
from GRU import GRUAutoencoder
from GRU_LSTM_Hybrid import GRU_LSTM_Autoencoder
from utils import plot_loss, reshape_patient_data

class AutoencoderTuner:
    def __init__(self,
                 data_path,
                 model='LSTM',  # Default to LSTM, can be 'GRU' or 'GRU_LSTM'
                 input_shape=(51, 2),
                 num_units=3,
                 save_dir=r'C:\Users\Admin\Desktop\Research paper\HP_results',
                 tuner_dir='tuner_dir',
                 max_trials=20,
                 executions_per_trial=3,
                 search_technique='bayesian',
                 use_attention=True,
                 epochs=60):
        
        self.data_path = data_path
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.epochs = epochs
        self.use_attention = use_attention
        self.num_units = num_units
        self.tuner_dir = tuner_dir
        self.model = model.lower()
        self.folder_name = self.save_dir + f'/{self.model.upper()}'

        attention_flag = 'attention' if self.use_attention else 'no_attention'
        project_name = f"{self.model}_{attention_flag}_autoencoder_{self.num_units}_units"


        # Load and preprocess data
        self.X_val = self._load_and_preprocess()

        # Initialize tuner
        if search_technique.lower() == 'bayesian':
            self.tuner = BayesianOptimization(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=tuner_dir,
                project_name=project_name
            )
        elif search_technique.lower() == 'random':
            self.tuner = RandomSearch(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=tuner_dir,
                project_name=project_name
            )
        elif search_technique.lower() == 'grid':
            self.tuner = GridSearch(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=tuner_dir,
                project_name=project_name
            )

    def build_model(self, hp):
        lstm_units = [hp.Int(f'units_{i}', min_value=16, max_value=164, step=8) for i in range(self.num_units)]
        gru_units = [hp.Int(f'units_{i}', min_value=16, max_value=164, step=8) for i in range(self.num_units)]
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.6, step=0.05)
        latent_dim = hp.Int('latent_dim', 64, 164, step=8)
        l2_reg = hp.Choice('l2_reg', values=[0.0, 1e-4, 1e-3])

        if self.model.lower() == 'lstm':
            model = LSTMAutoencoder(
                input_shape=self.input_shape,
                latent_dim=latent_dim,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                use_attention=self.use_attention
            ).model

        elif self.model.lower() == 'gru':
            model = GRUAutoencoder(
                input_shape=self.input_shape,
                latent_dim=latent_dim,
                gru_units=gru_units,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                use_attention=self.use_attention
            ).model
        
        elif self.model.lower() == 'gru_lstm':
            model = GRU_LSTM_Autoencoder(
                input_shape=self.input_shape,
                latent_dim=latent_dim,
                lstm_units=lstm_units,
                gru_units=gru_units,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                use_attention=self.use_attention
            ).model


        model.compile(optimizer='adam', loss='mse')
        return model

    def _load_and_preprocess(self):
        data = pd.read_csv(self.data_path)
        X = np.array(data.iloc[:, 1:])
        X, hip_scaler, knee_scaler = reshape_patient_data(X)
        print("Data shape after reshaping:", X.shape)
        return X

    

    def run_search(self):

        early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

        self.tuner.search(self.X_val, self.X_val, epochs=self.epochs, validation_split=0.2, callbacks=[early_stop])

        self.best_model = self.tuner.get_best_models(1)[0]
        self.best_hyperparameters = self.tuner.get_best_hyperparameters(1)[0]

        print("Best Hyperparameters:\n", self.best_hyperparameters.values)
        print(f"Best {self.model.upper()} Model Summary:",self.best_model.summary())

        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        # Save best hyperparameters to YAML
        best_hparams_dict = self.best_hyperparameters.values
        best_hparams_path = os.path.join(
            self.folder_name,
            f"{self.model.upper()}_best_hyperparameters_{'attention' if self.use_attention else 'no_attention'}_{self.num_units}_units.yaml"
        )

        with open(best_hparams_path, 'w') as f:
            yaml.dump(best_hparams_dict, f, default_flow_style=False)

        # Print tuner results
        self.tuner.results_summary()


    def save_results(self):
        model_path_h5 = os.path.join(self.folder_name, f"best_{self.model}_{'attention' if self.use_attention else 'no_attention'}_autoencoder_{self.num_units}_units_v1.h5")
        model_path_keras = os.path.join(self.folder_name, f"best_{self.model}_{'attention' if self.use_attention else 'no_attention'}_autoencoder_{self.num_units}_units_v1.keras")
        self.best_model.save(model_path_h5)
        self.best_model.save(model_path_keras)

        # Training again with early stopping to plot history
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        history = self.best_model.fit(
            self.X_val, self.X_val,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
        )

        plot_loss(history, label_1='Training Loss', label_2='Validation Loss')

        history_df = pd.DataFrame(history.history)
        history_csv = os.path.join(self.folder_name, f"val_training_history_LSTM_{'attention' if self.use_attention else 'no_attention'}_{self.num_units}_units.csv")
        history_df.to_csv(history_csv, index=False)




if __name__ == "__main__":
    tuner = AutoencoderTuner(
        data_path=r'C:\Users\Admin\Desktop\Research paper\Data\val.csv',
        num_units=4, 
        model='GRU'  # Change to 'GRU' or 'GRU_LSTM' as needed  
    )
    tuner.run_search()
    tuner.save_results()
