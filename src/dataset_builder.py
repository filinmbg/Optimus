import pandas as pd
import numpy as np
from src.feature_engineering import add_technical_indicators

def create_labeled_dataset(df: pd.DataFrame, window_size=50, pred_gap=10, target_pct=0.015):
    df = add_technical_indicators(df)
    df = df.reset_index(drop=True)

    sequences = []
    labels = []

    for i in range(len(df) - window_size - pred_gap):
        window = df.iloc[i:i + window_size]
        future_price = df.iloc[i + window_size + pred_gap - 1]['close']
        current_price = df.iloc[i + window_size - 1]['close']

        features = window.values

        change = (future_price - current_price) / current_price
        label = 1 if change >= target_pct else 0

        sequences.append(features)
        labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    return X, y

if __name__ == "__main__":
    df = pd.read_csv("data/BTCUSDT_15m.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    X, y = create_labeled_dataset(df)

    print("Форма X:", X.shape)
    print("Форма y:", y.shape)
    print("Кількість позитивних міток:", y.sum())
