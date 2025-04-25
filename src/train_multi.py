import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from src.dataset_builder_multi import build_multi_dataset
from src.model import build_lstm_model

def train_multi_model():
    X, y = build_multi_dataset()

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
        "models/optimus_multi_lstm.h5",
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
    train_multi_model()
