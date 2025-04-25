import pandas as pd
import numpy as np
from keras.models import load_model
from src.feature_engineering import add_technical_indicators
from src.dataset_builder import create_labeled_dataset

def backtest_multi_on_symbol(
    model_path="models/optimus_multi_lstm.h5",
    symbol="BTCUSDT",
    threshold=0.82,
    take_profit=0.015,  # TP 1.0%
    stop_loss=0.01,    # SL 1.0%
    initial_balance=1000,
    trade_pct=0.10,
    rsi_filter=True,
    rsi_threshold=40
):
    print(f"\n🔁 Backtest on {symbol} | Multi-trained | RSI < {rsi_threshold} | Threshold ≥ {threshold:.2f}")

    df = pd.read_csv(f"data/{symbol}_15m.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_with_indicators = add_technical_indicators(df.copy())
    df_raw = df.copy()

    X, y = create_labeled_dataset(df)
    model = load_model(model_path)
    preds = model.predict(X).flatten()

    results = []
    balance = initial_balance
    equity_curve = []
    initial_index = 50  # бо window = 50

    for i in range(len(preds)):
        if preds[i] >= threshold:
            entry_index = initial_index + i

            # 🎯 RSI фільтр
            if rsi_filter:
                rsi_value = df_with_indicators.iloc[entry_index]['rsi']
                if rsi_value >= rsi_threshold:
                    continue  # скип трейд, якщо RSI ≥ 40

            entry_price = df_raw.iloc[entry_index]['close']
            amount_usd = balance * trade_pct
            quantity = amount_usd / entry_price

            future_window = df_raw.iloc[entry_index+1:entry_index+11]
            max_price = future_window['high'].max()
            min_price = future_window['low'].min()

            tp_price = entry_price * (1 + take_profit)
            sl_price = entry_price * (1 - stop_loss)

            if max_price >= tp_price:
                exit_price = tp_price
                outcome = "TP"
            elif min_price <= sl_price:
                exit_price = sl_price
                outcome = "SL"
            else:
                exit_price = future_window.iloc[-1]['close']
                outcome = "EXIT"

            pnl_usd = (exit_price - entry_price) * quantity
            balance += pnl_usd

            results.append({
                "index": entry_index,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "usd_result": pnl_usd,
                "balance_after": balance,
                "outcome": outcome,
                "rsi": rsi_value,
                "pred": preds[i]
            })

            equity_curve.append(balance)

    total = len(results)
    wins = sum(1 for r in results if r["usd_result"] > 0)
    avg_pnl = np.mean([r["usd_result"] for r in results])
    final_balance = balance

    print(f"\n📈 Всього трейдів: {total}")
    print(f"✅ Виграшних: {wins} ({wins / total:.2%})")
    print(f"💰 Середній прибуток на трейд: ${avg_pnl:.2f}")
    print(f"📊 Кінцевий баланс: ${final_balance:.2f}")
    print(f"📈 Зростання: {((final_balance - initial_balance) / initial_balance) * 100:.2f}%")

    return results, equity_curve

if __name__ == "__main__":
    backtest_multi_on_symbol()
