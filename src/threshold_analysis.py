import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from src.dataset_builder import create_labeled_dataset
from src.feature_engineering import add_technical_indicators

def threshold_analysis(model_path="models/optimus_lstm.h5", symbol="BTCUSDT"):
    df = pd.read_csv(f"data/{symbol}_15m.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    X, y = create_labeled_dataset(df)

    model = load_model(model_path)
    preds = model.predict(X).flatten()

    print("Макс. значення передбачення:", np.max(preds))
    print("Розподіл ймовірностей:")
    print(pd.Series(preds).describe(percentiles=[.25, .5, .75, .9, .95, .99]))

    thresholds = np.arange(0.80, 0.96, 0.01)

    results = []

    for threshold in thresholds:
        total = 0
        correct = 0

        for i in range(len(preds)):
            if preds[i] >= threshold:
                total += 1
                if y[i] == 1:
                    correct += 1

        winrate = correct / total if total > 0 else 0
        results.append((threshold, total, winrate))

    return results


def plot_thresholds(results):
    thresholds = [r[0] for r in results]
    trades = [r[1] for r in results]
    winrates = [r[2] * 100 for r in results]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Winrate (%)', color=color)
    ax1.plot(thresholds, winrates, marker='o', color=color, label='Winrate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Кількість трейдів', color=color)
    ax2.plot(thresholds, trades, marker='x', color=color, label='Trades')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Winrate та Кількість трейдів на різних порогах')
    fig.tight_layout()
    plt.savefig("logs/threshold_analysis.png")
    plt.show()

if __name__ == "__main__":
    results = threshold_analysis()
    for t, n, w in results:
        print(f"Threshold: {t:.2f} | Трейдів: {n} | Winrate: {w:.2%}")
    plot_thresholds(results)
