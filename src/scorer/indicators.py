"""
技术指标计算模块 - 提供各种技术指标的计算函数
"""

import numpy as np
import pandas as pd
import talib as ta

def calculate_rsi(data, periods=[6, 14, 28]):
    """
    计算多周期RSI指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    periods (list): 需要计算的RSI周期列表
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    for period in periods:
        result[f'rsi_{period}'] = ta.RSI(result['close'], timeperiod=period)
    return result

def calculate_bollinger_bands(data, period=20, std_devs=[2, 3]):
    """
    计算布林带指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    period (int): 布林带周期
    std_devs (list): 标准差倍数列表
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    for std in std_devs:
        upper, middle, lower = ta.BBANDS(
            result['close'], timeperiod=period, 
            nbdevup=std, nbdevdn=std
        )
        suffix = '' if std == 2 else f'_{std}std'
        result[f'bb_upper{suffix}'] = upper
        result[f'bb_middle{suffix}'] = middle
        result[f'bb_lower{suffix}'] = lower
        
        # 计算布林带宽度 (作为波动率指标)
        result[f'bb_width{suffix}'] = (upper - lower) / middle * 100
    
    return result

def calculate_moving_averages(data, periods=[10, 20, 50, 200], types=['sma', 'ema']):
    """
    计算移动平均线指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    periods (list): 需要计算的周期列表
    types (list): 移动平均类型 ('sma', 'ema', 'wma')
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    for period in periods:
        if 'sma' in types:
            result[f'sma_{period}'] = ta.SMA(result['close'], timeperiod=period)
        if 'ema' in types:
            result[f'ema_{period}'] = ta.EMA(result['close'], timeperiod=period)
        if 'wma' in types:
            result[f'wma_{period}'] = ta.WMA(result['close'], timeperiod=period)
    
    return result

def calculate_volume_indicators(data, periods=[10, 20]):
    """
    计算成交量相关指标
    
    参数:
    data (pandas.DataFrame): 包含'close'和'volume'列的DataFrame
    periods (list): 需要计算的周期列表
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    # 成交量移动平均
    for period in periods:
        result[f'volume_sma_{period}'] = ta.SMA(result['volume'], timeperiod=period)
    
    # 计算相对成交量 (当日成交量/20日平均成交量)
    result['relative_volume'] = result['volume'] / result['volume_sma_20']
    
    # 成交量波动率 (20日成交量标准差/20日平均成交量)
    volumes = result['volume'].rolling(window=20).std()
    result['volume_volatility'] = volumes / result['volume_sma_20']
    
    # OBV (累积能量潮)
    result['obv'] = ta.OBV(result['close'], result['volume'])
    
    return result

def calculate_volatility_indicators(data):
    """
    计算波动率相关指标
    
    参数:
    data (pandas.DataFrame): 包含'high', 'low', 'close'列的DataFrame
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    # ATR - 平均真实波幅
    result['atr'] = ta.ATR(result['high'], result['low'], result['close'], timeperiod=14)
    
    # 相对ATR (ATR/收盘价)
    result['relative_atr'] = result['atr'] / result['close'] * 100
    
    # ATR变化率
    result['atr_change'] = result['atr'].pct_change(10) * 100
    
    return result

def calculate_momentum_indicators(data):
    """
    计算动量相关指标
    
    参数:
    data (pandas.DataFrame): 包含'high', 'low', 'close'列的DataFrame
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    # 随机指标
    result['slowk'], result['slowd'] = ta.STOCH(
        result['high'], result['low'], result['close'],
        fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )
    
    # Williams %R
    result['willr'] = ta.WILLR(result['high'], result['low'], result['close'], timeperiod=14)
    
    # CCI - 商品通道指数
    result['cci'] = ta.CCI(result['high'], result['low'], result['close'], timeperiod=20)
    
    # MACD
    result['macd'], result['macd_signal'], result['macd_hist'] = ta.MACD(
        result['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    return result

def identify_support_levels(data, lookback=100, n_points=2):
    """
    识别支撑位和斐波那契回调位
    
    参数:
    data (pandas.DataFrame): 包含'high', 'low', 'close'列的DataFrame
    lookback (int): 用于寻找支撑位的历史数据点数量
    n_points (int): 确定局部最低点的前后对比点数
    
    返回:
    tuple: (支撑位列表, 斐波那契回调位字典)
    """
    # 仅使用最近lookback个数据点来计算支撑阻力位
    recent_data = data.tail(lookback)
    
    # 寻找局部极小值作为支撑位
    support_levels = []
    for i in range(n_points, len(recent_data) - n_points):
        is_support = True
        for j in range(1, n_points + 1):
            if not (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-j] and 
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+j]):
                is_support = False
                break
        if is_support:
            support_levels.append(recent_data['low'].iloc[i])
    
    # 计算斐波那契回调位
    fib_levels = {}
    if len(recent_data) > 0:
        max_price = recent_data['high'].max()
        min_price = recent_data['low'].min()
        price_range = max_price - min_price
        
        fib_levels = {
            'fib_0': min_price,
            'fib_0.236': min_price + 0.236 * price_range,
            'fib_0.382': min_price + 0.382 * price_range,
            'fib_0.5': min_price + 0.5 * price_range,
            'fib_0.618': min_price + 0.618 * price_range,
            'fib_0.786': min_price + 0.786 * price_range,
            'fib_1': max_price
        }
    
    return support_levels, fib_levels

def identify_candlestick_patterns(data):
    """
    识别常见蜡烛图形态
    
    参数:
    data (pandas.DataFrame): 包含'open', 'high', 'low', 'close'列的DataFrame
    
    返回:
    pandas.DataFrame: 包含识别结果的原始DataFrame
    """
    result = data.copy()
    
    # 添加蜡烛图相关属性
    result['body_size'] = abs(result['close'] - result['open'])
    result['total_range'] = result['high'] - result['low']
    result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
    result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
    
    # 长下影线 (潜在反转信号)
    result['long_lower_shadow'] = (result['lower_shadow'] > 2 * result['body_size'])
    
    # 锤子线形态
    result['hammer'] = (
        (result['lower_shadow'] > 1.5 * result['body_size']) & 
        (result['body_size'] < 0.3 * result['total_range'])
    )
    
    # 十字星形态 (不确定性)
    result['doji'] = (result['body_size'] < 0.1 * result['total_range'])
    
    return result

def detect_divergences(data, lookback=20):
    """
    检测RSI和MACD与价格的背离
    
    参数:
    data (pandas.DataFrame): 包含指标计算结果的DataFrame
    lookback (int): 用于检测背离的历史数据点数量
    
    返回:
    pandas.DataFrame: 包含背离检测结果的原始DataFrame
    """
    result = data.copy()
    
    # 初始化背离列
    result['rsi_bullish_divergence'] = False
    result['macd_bullish_divergence'] = False
    result['obv_bullish_divergence'] = False
    
    # 检测每个点是否存在背离
    for i in range(lookback, len(result)):
        # 确定当前窗口
        window = result.iloc[i-lookback:i+1]
        
        # 获取当前价格和前低点
        current_price = window['close'].iloc[-1]
        prev_low_price = window['close'].iloc[:-1].min()
        
        # 价格创新低
        if current_price < prev_low_price:
            # RSI底背离
            current_rsi = window['rsi_14'].iloc[-1]
            prev_low_rsi = window['rsi_14'].iloc[:-1].min()
            if current_rsi > prev_low_rsi:
                result.loc[result.index[i], 'rsi_bullish_divergence'] = True
            
            # MACD底背离
            current_macd_hist = window['macd_hist'].iloc[-1]
            prev_low_macd_hist = window['macd_hist'].iloc[:-1].min()
            if current_macd_hist > prev_low_macd_hist:
                result.loc[result.index[i], 'macd_bullish_divergence'] = True
            
            # OBV底背离
            current_obv = window['obv'].iloc[-1]
            prev_low_obv = window['obv'].iloc[:-1].min()
            if current_obv > prev_low_obv:
                result.loc[result.index[i], 'obv_bullish_divergence'] = True
    
    return result