import pandas as pd
import numpy as np
from keras.models import load_model
from src.feature_engineering import add_technical_indicators
from src.dataset_builder import create_labeled_dataset

def backtest_with_pnl(
    model_path="models/optimus_lstm.h5",
    symbol="BTCUSDT",
    threshold=0.90,
    take_profit=0.01,  # 2.5%
    stop_loss=0.02,     # 2%
    initial_balance=1000,
    trade_pct=0.10      # 10% балансу на трейд
):
    print(f"📊 Backtest with PnL for {symbol} | Threshold: {threshold:.2f} | TP: {take_profit*100:.1f}% | SL: {stop_loss*100:.1f}% | Start balance: ${initial_balance}")

    df = pd.read_csv(f"data/{symbol}_15m.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_raw = df.copy()

    X, y = create_labeled_dataset(df)
    model = load_model(model_path)
    preds = model.predict(X).flatten()

    results = []
    balance = initial_balance
    equity_curve = []

    initial_index = 50

    for i in range(len(preds)):
        if preds[i] >= threshold:
            entry_index = initial_index + i
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
                "pred": preds[i]
            })

            equity_curve.append(balance)

    total = len(results)
    wins = sum(1 for r in results if r["usd_result"] > 0)
    avg_pnl = np.mean([r["usd_result"] for r in results])
    final_balance = balance

    print(f"📈 Всього трейдів: {total}")
    print(f"✅ Виграшних: {wins} ({wins / total:.2%})")
    print(f"💰 Середній прибуток на трейд: ${avg_pnl:.2f}")
    print(f"📊 Кінцевий баланс: ${final_balance:.2f}")
    print(f"📈 Зростання: {((final_balance - initial_balance) / initial_balance) * 100:.2f}%")

    return results, equity_curve

if __name__ == "__main__":
    backtest_with_pnl()
