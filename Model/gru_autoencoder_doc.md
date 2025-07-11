
# 📘 Documentation: GRU Autoencoder with Optional Attention

This module implements a GRU-based autoencoder for sequential data (e.g., joint angle time-series), optionally enhanced with an attention mechanism.

---

## 📦 Imports

The script uses:
- `TensorFlow` and `Keras` – model creation and training
- `scikit-learn` – preprocessing and metrics
- `NumPy`, `Pandas`, `Matplotlib` – data handling and visualization

---

## 🔍 AttentionLayer Class

### `AttentionLayer(tf.keras.layers.Layer)`
A custom Keras layer that performs **scaled dot-product self-attention** over time.

- **Purpose**: Lets the network focus on important time steps in the sequence.
- **Returns**: Attention-weighted context representation.

---

## 🧠 GRUAutoencoder Class

### `GRUAutoencoder(...)`
Builds, compiles, and trains a GRU-based sequence autoencoder with optional attention.

### **Parameters:**
- `input_shape` *(tuple)*: Shape of input sequences (default: (51, 2))
- `latent_dim` *(int)*: Size of the latent vector
- `gru_units` *(list)*: Units in each GRU layer of the encoder/decoder
- `dropout_rate` *(float)*: Dropout rate for regularization
- `l2_reg` *(float)*: L2 regularization strength
- `use_attention` *(bool)*: Whether to include attention layers

---

## 🔧 Methods

### `build_model()`
Creates the full GRU autoencoder.

- **Encoder**:
  - Stacked GRU layers
  - Optional attention
  - Dense latent layer

- **Decoder**:
  - Repeats the latent vector
  - Stacked GRU layers (in reverse order)
  - Optional attention
  - Final TimeDistributed Dense output layer

**Returns**: A Keras `Model` object for the full autoencoder and a separate encoder model.

---

### `compile(*args, **kwargs)`
Compiles the autoencoder model using provided loss, optimizer, etc.

---

### `fit(*args, **kwargs)`
Trains the autoencoder model on input data.

---

### `predict(X)`
Returns reconstructed sequences for input `X`.

---

### `latent_representation(X)`
Returns the latent encoded representation of input sequences using the encoder.

---

### `save(model_path='gru_autoencoder_model.h5')`
Saves the full autoencoder model to a file.

---

## ✅ Example Usage

```python
model = GRUAutoencoder(input_shape=(51, 2), use_attention=True)
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=50, validation_split=0.2)
reconstructed = model.predict(X_test)
latent = model.latent_representation(X_test)
model.save("gru_autoencoder_with_attention.h5")
```

Ideal for:
- Sequence reconstruction tasks
- Latent feature extraction for clustering
- Time-series anomaly detection
