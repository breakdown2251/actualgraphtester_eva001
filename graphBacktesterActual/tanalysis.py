import numpy as np

def rsi(close, length):
    """
    Calculates the Relative Strength Index (RSI) of a given series.

    Parameters:
        close (pandas.Series): Close prices.
        length (int): Length of the RSI calculation window.

    Returns:
        pandas.Series: RSI values.
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def hma(series, length):
    """
    Calculates the Hull Moving Average (HMA) of a given series.

    Parameters:
        series (pandas.Series): Series to calculate HMA for.
        length (int): Length of the HMA calculation window.

    Returns:
        pandas.Series: HMA values.
    """
    wma1 = 2 * series.rolling(window=length // 2).mean() - series.rolling(window=length).mean()
    wma2 = np.sqrt(length) * (series - series.rolling(window=length // 2).mean()).rolling(window=length).mean()
    hma = wma2.rolling(window=int(np.sqrt(length))).mean()
    return hma

def ema(series, length):
    """
    Calculates the Exponential Moving Average (EMA) of a given series.

    Parameters:
        series (pandas.Series): Series to calculate EMA for.
        length (int): Length of the EMA calculation window.

    Returns:
        pandas.Series: EMA values.
    """
    return series.ewm(span=length, adjust=False).mean()

def ma(series, length, ma_type='simple'):
    """
    Calculates the Moving Average (MA) of a given series.

    Parameters:
        series (pandas.Series): Series to calculate MA for.
        length (int): Length of the MA calculation window.
        ma_type (str): Type of moving average ('simple', 'exponential', etc.).

    Returns:
        pandas.Series: MA values.
    """
    if ma_type == 'simple':
        return series.rolling(window=length).mean()
    elif ma_type == 'exponential':
        return series.ewm(span=length, adjust=False).mean()
    # Add more MA types as needed

def valuewhen(condition, series, offset):
    """
    Returns the value of the series when the condition is met, with a specified offset.

    Parameters:
        condition (bool): Condition to be met.
        series (pandas.Series): Series to retrieve values from.
        offset (int): Offset from the condition index.

    Returns:
        pandas.Series: Values of the series when the condition is met, with the specified offset.
    """
    return series.shift(offset).where(condition)

def pivotlow(series, leftbars, rightbars):
    """
    Function to identify pivot lows in a series.

    Args:
    - series: Pandas Series or array-like object.
    - leftbars: Integer representing the number of bars to the left for comparison.
    - rightbars: Integer representing the number of bars to the right for comparison.

    Returns:
    - pandas.Series: A boolean Series indicating True where pivot lows are found, False otherwise.
    """
    pivot_lows = (series.shift(-leftbars) > series) & (series.shift(rightbars) > series)
    return pivot_lows

def pivothigh(series, leftbars, rightbars):
    """
    Function to identify pivot highs in a series.

    Args:
    - series: Pandas Series or array-like object.
    - leftbars: Integer representing the number of bars to the left for comparison.
    - rightbars: Integer representing the number of bars to the right for comparison.

    Returns:
    - pandas.Series: A boolean Series indicating True where pivot highs are found, False otherwise.
    """
    pivot_highs = (series.shift(-leftbars) < series) & (series.shift(rightbars) < series)
    return pivot_highs
