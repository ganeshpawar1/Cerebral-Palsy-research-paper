
# 📘 Documentation: LSTM Autoencoder with Optional Attention

This module implements an LSTM-based autoencoder for sequence data (e.g., joint angle time-series), optionally using an attention mechanism.

---

## 📦 Imports

The code relies on the following key packages:
- `TensorFlow` and `Keras` – deep learning framework
- `scikit-learn` – preprocessing and evaluation
- `NumPy`, `Pandas`, `Matplotlib` – data manipulation and visualization

---

## 🔍 AttentionLayer Class

### `AttentionLayer(tf.keras.layers.Layer)`
Custom Keras layer implementing **scaled dot-product self-attention** over time steps.

- **Purpose**: Enhances sequence modeling by allowing the network to attend to relevant parts of the input sequence.
- **Returns**: Attention-weighted context vectors.

---

## 🧠 LSTMAutoencoder Class

### `LSTMAutoencoder(...)`
A class to create, compile, and train an LSTM autoencoder with optional attention.

### **Parameters:**
- `input_shape` *(tuple)*: Shape of the input sequence (default: (51, 2))
- `latent_dim` *(int)*: Dimensionality of the latent vector
- `lstm_units` *(list)*: List of LSTM units in each encoder layer
- `dropout_rate` *(float)*: Dropout for regularization
- `l2_reg` *(float)*: L2 regularization factor
- `use_attention` *(bool)*: Whether to use the Attention layer

---

## 🔧 Methods

### `build_model()`
Builds the encoder-decoder architecture using LSTM and optional Attention layers.

- **Encoder**:
  - Stacked LSTM layers
  - Optional self-attention
  - Dense latent layer

- **Decoder**:
  - Repeat vector
  - Reversed LSTM layers
  - Optional self-attention
  - Final time-distributed dense layer

**Returns**: Compiled Keras `autoencoder` and `encoder` models.

---

### `compile(*args, **kwargs)`
Calls `self.model.compile()` with given arguments (like loss function, optimizer).

---

### `fit(*args, **kwargs)`
Calls `self.model.fit()` to train the model.

---

### `predict(X)`
Performs prediction (reconstruction) on input data `X`.

---

### `latent_representation(X)`
Returns latent representation of input `X` using the encoder.

---

### `save(model_path='autoencoder_model.h5')`
Saves the full model to an H5 file.

---

## ✅ Example Usage

```python
ae = LSTMAutoencoder(input_shape=(51,2), use_attention=True)
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_train, X_train, epochs=50, validation_split=0.2)
recon = ae.predict(X_test)
latent = ae.latent_representation(X_test)
ae.save("model_attention.h5")
```

This setup is useful for tasks like:
- Gait cycle reconstruction
- Anomaly detection
- Latent feature extraction
