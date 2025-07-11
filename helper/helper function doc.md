
# 📘 Documentation: Gait Joint Data Utility and Visualization Functions

This module contains utility functions for:
- Visualizing model training and reconstruction performance
- Preprocessing joint data (hip and knee angles)
- Evaluating model reconstruction errors

---

## 🖼️ 1. Plotting Functions

### `plot_loss(history, label_1='Training Loss', label_2='Validation Loss', save_path='')`
- Plots training and validation loss over epochs.
- **Parameters:**
  - `history`: History object returned by Keras `model.fit()`
  - `label_1`, `label_2`: Custom labels for the plots
  - `save_path`: If provided, saves the plot as an image

### `plot_histogram(data, bins=30, xlabel='Value', ylabel='Frequency', title='Histogram', save_path='')`
- Draws a histogram for given data.
- Useful for error or value distribution visualization.

### `plot_reconstruction_graph(X_test, X_pred, idx_range=1, y_label='Angle', save_path='')`
- Plots **actual vs reconstructed** hip and knee joint angles for selected test samples.
- **Parameters:**
  - `X_test`, `X_pred`: Input and predicted sequences, shape `(samples, frames, 2)`
  - `idx_range`: Number of samples to plot
  - `y_label`: Y-axis label (e.g., "Angle")
  - `save_path`: Folder to save plots (optional)

### `plot_reconstruction_error(X_test, X_pred, save_path='')`
- Plots **MSE and MAE per test sample**.
- Helps visualize which samples were reconstructed poorly.

---

## 🔧 2. Preprocessing Functions

### `normalize_joint(joint_data)`
- Normalizes data using Min-Max scaling (0 to 1).
- **Returns**: Scaled data and the `MinMaxScaler` used.

### `reshape_patient_data(X)`
- Prepares raw joint data for model input.
- **Input**: 2D array `X` of shape `(samples, 102)`  
  - First 51: Hip angles  
  - Last 51: Knee angles
- **Steps**:
  - Splits data into hip and knee
  - Imputes missing values using the mean
  - Normalizes both joints
  - Stacks them → `(samples, 51, 2)`
- **Returns**: Stacked data, hip scaler, knee scaler

### `denormalize(joint_data, hip_scaler, knee_scaler)`
- Reverses normalization on joint data.
- **Input**: Normalized joint data of shape `(samples, 51, 2)`
- **Output**: Denormalized data in same shape

---

## 📊 3. Evaluation Function

### `model_error_info(X_test, X_pred, save_path='', model_name='model_x')`
- Computes detailed error metrics between actual and predicted values.
- **Outputs**:
  - Mean, median, std of MSE and MAE
  - Top and bottom 10% and 25% error segments
  - Frame-wise MAE
  - Error in specific frame intervals (0–10, 10–30, 30–50)
  - Worst performing frame and its error
- **Optional**: Saves summary as a CSV file
- **Returns**: A dictionary of metrics

---

## 📦 Dependencies
- `matplotlib` – for plotting
- `numpy` – numerical operations
- `pandas` – saving results
- `sklearn` – preprocessing and error metrics

---

## ✅ Example Use Cases
- Track training loss of an autoencoder model for joint angle prediction.
- Check how well your model reconstructs real gait data.
- Identify which frames/joint movements are hardest to predict.
