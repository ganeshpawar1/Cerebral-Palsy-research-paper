import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_loss(history,label_1='Training Loss', label_2='Validation Loss',save_path=''):
    plt.plot(history.history['loss'], label=label_1)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label=label_2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training vs Validation Loss')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_histogram(data, bins=30, xlabel='Value', ylabel='Frequency', title='Histogram',save_path=''):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path)  
    plt.show()

def plot_reconstruction_graph(X_test,X_pred,idx_range=1,y_label='Angle',save_path=''):
    if idx_range>len(X_test):
        idx_range = len(X_test)

    for idx in range(idx_range):
        actual = X_test[idx]
        reconstructed = X_pred[idx]
        plt.figure(figsize=(10, 5))
        plt.plot(actual[:, 0], label='Actual Hip', color='blue')
        plt.plot(reconstructed[:, 0], label='Reconstructed Hip', linestyle='--', color='blue')
        plt.plot(actual[:, 1], label='Actual Knee', color='green')
        plt.plot(reconstructed[:, 1], label='Reconstructed Knee', linestyle='--', color='green')
        plt.title('Actual vs Reconstructed Knee and Hip Angles')
        plt.xlabel('Frame')
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        if save_path is not '':
            plt.savefig(save_path + fr'\reconstruct_graph_idx_{idx}.png')
        plt.close() 


def plot_reconstruction_error(X_test, X_pred, save_path=''):
    mse_list = [mean_squared_error(X_test[i], X_pred[i]) for i in range(len(X_test))]
    mae_list = [mean_absolute_error(X_test[i], X_pred[i]) for i in range(len(X_test))]
    plt.figure(figsize=(10, 4))
    plt.plot(mse_list, label='Test MSE per sample')
    plt.plot(mae_list, label='Test MAE per sample')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Error')
    plt.title('Reconstruction Error on Test Set')
    plt.legend()
    plt.tight_layout()
    if save_path != '':
        plt.savefig(save_path + r'\test_reconstruction_error.png')
    plt.show()

def normalize_joint(joint_data):
    scaler = MinMaxScaler()
    joint_data_scaled = scaler.fit_transform(joint_data)
    return joint_data_scaled, scaler

def model_error_info(X_test, X_pred, save_path='', model_name='model_x'):
    n_samples, n_frames, _ = X_test.shape

    mse_errors = np.array([mean_squared_error(X_test[i], X_pred[i]) for i in range(n_samples)])
    mae_errors = np.array([mean_absolute_error(X_test[i], X_pred[i]) for i in range(n_samples)])

    # Sorting for top/bottom samples
    sorted_mse = np.sort(mse_errors)
    sorted_mae = np.sort(mae_errors)

    result = {
        'model': model_name,
        'mse_mean': np.mean(mse_errors),
        'mse_median': np.median(mse_errors),
        'mse_std': np.std(mse_errors),
        'mse_top_10%': np.mean(sorted_mse[-int(0.1 * n_samples):]),
        'mse_top_25%': np.mean(sorted_mse[-int(0.25 * n_samples):]),
        'mse_bottom_10%': np.mean(sorted_mse[:int(0.1 * n_samples)]),
        'mse_bottom_25%': np.mean(sorted_mse[:int(0.25 * n_samples)]),

        'mae_mean': np.mean(mae_errors),
        'mae_median': np.median(mae_errors),
        'mae_std': np.std(mae_errors),
        'mae_top_10%': np.mean(sorted_mae[-int(0.1 * n_samples):]),
        'mae_top_25%': np.mean(sorted_mae[-int(0.25 * n_samples):]),
        'mae_bottom_10%': np.mean(sorted_mae[:int(0.1 * n_samples)]),
        'mae_bottom_25%': np.mean(sorted_mae[:int(0.25 * n_samples)]),
    }

    # --- Frame-wise error ---
    frame_mae = np.mean(np.abs(X_test - X_pred), axis=(0, 2))  # shape: (n_frames,)
    framewise_result = {
        f'mae_frame_{i}': frame_mae[i] for i in range(n_frames)
    }

    # --- Interval-wise error ---
    frame_intervals = [(0, 10), (10, 30), (30, 50)]
    for (start, end) in frame_intervals:
        result[f'mae_{start}-{end}'] = np.mean(frame_mae[start:end])

    # --- Frame with max error ---
    result['worst_frame'] = int(np.argmax(frame_mae))
    result['max_frame_error'] = float(np.max(frame_mae))

    # --- Combine all info ---
    full_result = {**result, **framewise_result}

    # Optional save
    if save_path:
        df = pd.DataFrame([full_result])
        df.to_csv(f"{save_path}/{model_name}_error_summary.csv", index=False)

    return full_result
    

def reshape_patient_data(X):
    # Assume X has shape (n_samples, 102)
    hip = X[:, :51]   # first 51 features
    knee = X[:, 51:]  # next 51 features

    print("before hip&knee nan", np.any(np.isnan(hip)), np.any(np.isinf(hip)), 
          np.any(np.isnan(knee)), np.any(np.isinf(knee)))
    
    imputer = SimpleImputer(strategy='mean')  # or 'median', 'constant', etc.
    hip = imputer.fit_transform(hip)  # Apply on entire dataset before split
    knee = imputer.fit_transform(knee)  # Apply on entire dataset before split

    print("after hip&knee nan", np.any(np.isnan(hip)), np.any(np.isinf(hip)), 
          np.any(np.isnan(knee)), np.any(np.isinf(knee)))

    hip,hip_scaler = normalize_joint(hip)  # Normalize hip features
    knee,knee_scaler = normalize_joint(knee)  # Normalize knee features
    # Stack hip and knee features along the last axis → shape: (n_samples, 51, 2)
    return np.stack((hip, knee), axis=-1), hip_scaler, knee_scaler

def denormalize(joint_data, hip_scaler, knee_scaler):
    """
    joint_data: shape (n_samples, 51, 2)
    Returns:
        hip_original: shape (n_samples, 51)
        knee_original: shape (n_samples, 51)
    """
    hip_scaled = joint_data[:, :, 0]
    knee_scaled = joint_data[:, :, 1]
    
    hip_original = hip_scaler.inverse_transform(hip_scaled)
    knee_original = knee_scaler.inverse_transform(knee_scaled)

    return np.stack((hip_original, knee_original), axis=-1)

