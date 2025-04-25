import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Volume ratio
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume_20']

    # ATR
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']  # відносна ширина

    # Зміна ціни
    df['pct_change_1'] = df['close'].pct_change(1)
    df['pct_change_3'] = df['close'].pct_change(3)
    df['pct_change_5'] = df['close'].pct_change(5)

    # Свічкова динаміка
    df['candle_ratio'] = df['close'] / df['open']  # bullish > 1

    # Внутрішньо-свічкова волатильність
    df['bar_range'] = (df['high'] - df['low']) / df['close']

    # Прибрати рядки з NaN
    df = df.dropna().reset_index(drop=True)

    # Видаляємо зайві колонки
    df = df.drop(columns=[
        'timestamp', 'avg_volume_20', 'bb_high', 'bb_low'
    ], errors='ignore')

    # Масштабування
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df
