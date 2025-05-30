import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from src.model import build_lstm_model
from src.dataset_builder import create_labeled_dataset

def load_data():
    df = pd.read_csv("data/BTCUSDT_15m.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    X, y = create_labeled_dataset(df)
    return X, y

def train():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    checkpoint = ModelCheckpoint(
        "models/optimus_lstm.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
        class_weight=class_weights_dict
    )

    return model, history

if __name__ == "__main__":
    model, history = train()
