
# 📘 `TestLSTM` - LSTM Autoencoder Evaluation Script

This module provides functionality to load a trained LSTM autoencoder model, apply it to test data, compute reconstruction errors, and visualize the reconstruction performance.

---

## 📂 File Structure

```plaintext
Research paper/
├── Data/
│   └── test.csv
├── Model/
│   └── LSTM.py               # Contains LSTM model class
├── Training/
│   └── Model/
│       └── lstm_autoencoder_model_4_units_v1.keras
├── helper/
│   └── utils.py              # Contains helper functions: reshape, plot, denormalize
└── Testing/
    └── plots/                # Output plots will be saved here
```

---

## 🚀 Usage

### Run the script:

```bash
python test_lstm.py
```

> Make sure all custom paths and module dependencies are correct before running.

---

## 🧠 Class: `TestLSTM`

### Description

This class loads a trained LSTM autoencoder model, evaluates it on a provided test dataset, and generates plots of the reconstruction error and prediction performance.

### Parameters

| Argument     | Type   | Description                                                                 |
|--------------|--------|-----------------------------------------------------------------------------|
| `model_path` | `str`  | Path to the saved `.keras` LSTM model.                                      |
| `data_path`  | `str`  | Path to the `.csv` file containing test data.                                |
| `save_path`  | `str`  | Directory where plots will be saved.                                         |
| `idx_range`  | `int`  | Number of samples to visualize in reconstruction plots.                      |
| `model`      | `str`  | Name of the model folder (default `'LSTM'`).                                 |
| `units`      | `int`  | Number of LSTM units used in model (used for naming folders).                |

---

## 📦 Methods

### `load_data()`

- Loads test data from CSV.
- Applies reshaping and scaling using `reshape_patient_data`.
- Returns reshaped data and scalers for denormalization.

---

### `evaluate()`

- Loads the LSTM model.
- Predicts the reconstructed sequences.
- Computes reconstruction errors.
- Plots:
  - Histogram of reconstruction errors.
  - Scaled vs. predicted values.
  - Denormalized (original angle) reconstruction vs. ground truth.

#### Returns:
- `reconstruction_error`: `np.ndarray`, per-sample reconstruction MSE.
- `X`: Ground truth test input (scaled).
- `X_pred`: Model reconstruction output (scaled).

---

## 📊 Output

After running, the following will be saved:

- `plots/reconstruction_error_hist.png`: Histogram of reconstruction errors.
- `plots/LSTM/model_4_units/*.png`: Reconstruction graphs for selected samples (scaled & denormalized).

---

## 🧩 Dependencies

Ensure these are installed:

```bash
pip install numpy pandas tensorflow matplotlib
```

Also ensure local modules exist:
- `LSTM.py` contains the `LSTMAutoencoder` class.
- `utils.py` contains:
  - `reshape_patient_data`
  - `plot_reconstruction_error`
  - `plot_reconstruction_graph`
  - `denormalize`

---

## ✏️ Notes

- `compile=False` in `load_model()` is used since we're only doing inference.
- Only the columns after the first column in the CSV are considered as input features.
- Plotting utilities are assumed to support `save_path` argument.

---

## 📧 Author

Script written for testing trained LSTM autoencoders in the context of motion/gait data (e.g., CP patient joint angles).
