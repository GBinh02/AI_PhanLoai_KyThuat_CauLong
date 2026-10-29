"""5 kien truc ung vien (TensorFlow/Keras).

Neu truyen norm_data (tap train da qua build_features), model co san lop
Normalization (z-score theo tung dac trung, hoc tu train) nen model luu ra
dung doc lap, khong can luu rieng mean/std.
"""
from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAMES = ["mlp", "lstm", "bilstm", "gru", "cnn1d"]


def _head(x, n_classes):
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def build_model(name, input_shape, n_classes=4, norm_data=None):
    inp = keras.Input(shape=input_shape)
    x = inp
    if norm_data is not None:
        norm = layers.Normalization(axis=-1)
        norm.adapt(norm_data)
        x = norm(x)

    if name == "mlp":  # khong dung thu tu thoi gian, lam moc don gian nhat
        h = layers.Flatten()(x)
        h = layers.Dense(128, activation="relu")(h)
        h = layers.Dropout(0.4)(h)
        h = layers.Dense(64, activation="relu")(h)
        h = layers.Dropout(0.3)(h)
        out = layers.Dense(n_classes, activation="softmax")(h)
    elif name == "lstm":
        out = _head(layers.LSTM(64)(x), n_classes)
    elif name == "bilstm":
        out = _head(layers.Bidirectional(layers.LSTM(64))(x), n_classes)
    elif name == "gru":
        out = _head(layers.GRU(64)(x), n_classes)
    elif name == "cnn1d":
        h = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        h = layers.BatchNormalization()(h)
        h = layers.MaxPooling1D(2)(h)
        h = layers.Conv1D(128, 3, padding="same", activation="relu")(h)
        h = layers.BatchNormalization()(h)
        out = _head(layers.GlobalAveragePooling1D()(h), n_classes)
    else:
        raise ValueError(f"Model khong ton tai: {name}. Chon trong {MODEL_NAMES}")
    return keras.Model(inp, out, name=name)
