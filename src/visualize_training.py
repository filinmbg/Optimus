import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
import os


def plot_history(history):
    # Якщо історія передається як обʼєкт
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.grid()
    plt.savefig("logs/accuracy.png")

    # Loss
    plt.figure()
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid()
    plt.savefig("logs/loss.png")
    plt.show()


if __name__ == "__main__":
    from src.train import train

    print("Повторне тренування моделі для отримання історії...")
    model, history = train()

    print("Побудова графіків...")
    os.makedirs("logs", exist_ok=True)
    plot_history(history)
