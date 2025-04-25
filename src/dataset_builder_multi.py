import pandas as pd
import numpy as np
from src.feature_engineering import add_technical_indicators
import os

MONETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

def create_labeled_dataset(df: pd.DataFrame, window_size=50, pred_gap=10, target_pct=0.01):
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

    return np.array(sequences), np.array(labels)

def build_multi_dataset():
    X_all, y_all = [], []

    for symbol in MONETS:
        path = f"data/{symbol}_15m.csv"
        if not os.path.exists(path):
            print(f"⚠️ Пропущено {symbol} — файл не знайдено.")
            continue

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        X, y = create_labeled_dataset(df)
        X_all.append(X)
        y_all.append(y)
        print(f"✅ {symbol} → {X.shape[0]} трейдів")

    X_full = np.concatenate(X_all, axis=0)
    y_full = np.concatenate(y_all, axis=0)

    print(f"\n📦 Обʼєднано: {X_full.shape[0]} трейдів | Класів '1': {np.sum(y_full)} ({np.mean(y_full)*100:.2f}%)")
    return X_full, y_full

if __name__ == "__main__":
    X, y = build_multi_dataset()
    print("📐 X shape:", X.shape)
    print("🎯 y shape:", y.shape)
