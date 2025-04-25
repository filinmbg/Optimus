import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from src.feature_engineering import add_technical_indicators
from src.dataset_builder import create_labeled_dataset

def backtest_model(model_path="models/optimus_lstm.h5", symbol="BTCUSDT", file_path=None):
    print(f"📊 Backtest on: {symbol}")

    if file_path is None:
        file_path = f"data/{symbol}_15m.csv"

    # 1. Завантаження даних
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Побудова датасету
    window_size = 50
    pred_gap = 10
    target_pct = 0.02
    X, y = create_labeled_dataset(df, window_size, pred_gap, target_pct)

    # 3. Завантаження моделі
    model = load_model(model_path)

    # 4. Прогнозування
    preds = model.predict(X)
    preds = preds.flatten()

    # 5. Логіка входу у трейд
    results = []
    for i in range(len(preds)):
        if preds[i] > 0.8:
            label = y[i]
            results.append({
                "index": i,
                "pred": preds[i],
                "actual": label
            })

    # 6. Аналіз
    total = len(results)
    correct = sum([1 for r in results if r["actual"] == 1])
    winrate = correct / total if total > 0 else 0

    print(f"🔁 Трейдів здійснено: {total}")
    print(f"✅ Виграшних: {correct}")
    print(f"📈 Winrate: {winrate:.2%}")

    return results

if __name__ == "__main__":
    backtest_model()
