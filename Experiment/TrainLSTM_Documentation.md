
# 🏋️‍♂️ `TrainLSTM` - LSTM Autoencoder Training Script

This module provides functionality to load and preprocess training data, build and train an LSTM autoencoder model, and save both the model and training loss plots.

---

## 📂 File Structure

```plaintext
Research paper/
├── Data/
│   └── train.csv
├── Model/
│   └── LSTM.py                   # Contains LSTM model class definition
├── Training/
│   ├── Model/
│   │   └── lstm_autoencoder_model_4_units_v1.keras
│   └── loss plots/
│       └── training_loss_plot_4_units_v1.png
├── helper/
│   └── utils.py                  # Contains reshape and plotting utilities
├── HP_results/
│   └── LSTM/
│       └── best_lstm_attention_autoencoder_4_units_v1.h5
```

---

## 🚀 Usage

### Run the script:

```bash
python train_lstm.py
```

> Ensure file paths and module imports are set correctly before execution.

---

## 🧠 Class: `TrainLSTM`

### Description

Handles the entire training pipeline for an LSTM autoencoder model, including data loading, reshaping, training, and saving results.

### Parameters

| Argument         | Type   | Description                                                                 |
|------------------|--------|-----------------------------------------------------------------------------|
| `data_path`      | `str`  | Path to the `.csv` training data file.                                      |
| `epochs`         | `int`  | Number of training epochs (default: 30).                                    |
| `batch_size`     | `int`  | Batch size for training (default: 32).                                      |
| `model_path`     | `str`  | Path to a pretrained model (optional, not used in training here).           |
| `save_path`      | `str`  | Path to save the trained model.                                             |
| `plot_save_path` | `str`  | Path to save the training loss plot.                                        |

---

## 📦 Methods

### `load_data()`

- Loads and reshapes training data using `reshape_patient_data`.
- Returns the reshaped array along with hip and knee scalers.

---

### `train_model()`

- Splits reshaped data into training and validation sets.
- Initializes the LSTM autoencoder.
- Compiles the model using Mean Squared Error (MSE) loss and Adam optimizer.
- Trains the model.
- Saves training loss plot and model to the specified paths.

---

## 📊 Output

- Trained model file:  
  `Training/Model/lstm_autoencoder_model_4_units_v1.keras`

- Training loss plot:  
  `Training/loss plots/training_loss_plot_4_units_v1.png`

---

## 🧩 Dependencies

Ensure these Python packages are installed:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

Also ensure local modules exist:
- `LSTM.py` contains the `LSTMAutoencoder` class.
- `utils.py` contains:
  - `reshape_patient_data`
  - `plot_loss`

---

## ✏️ Notes

- NaN or Inf checks are performed to prevent training on corrupted data.
- Train/Validation split is 70/30 by default.
- Logging is included for dimensions, shapes, and NaN presence.

---

## 📧 Author

Script written for training LSTM autoencoders on joint angle datasets (e.g., hip/knee motion data).

