"""
技术指标计算模块 - 提供各种技术指标的计算函数，包括TA-Lib外部依赖和内部实现
"""

import numpy as np
import pandas as pd
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入TA-Lib
_talib_available = False
try:
    import talib as ta
    _talib_available = True
    logger.info("成功加载TA-Lib库")
except ImportError:
    logger.warning("未安装TA-Lib库，将使用内部实现的技术指标计算")

def has_talib():
    """
    检查是否可以使用TA-Lib
    
    返回:
    bool: 如果可以使用TA-Lib则为True
    """
    return _talib_available

def calculate_rsi(data, periods=[6, 14, 28], use_talib=True):
    """
    计算多周期RSI指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    periods (list): 需要计算的RSI周期列表
    use_talib (bool): 是否使用TA-Lib库计算，如果不可用则使用内部实现
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        for period in periods:
            if use_talib and _talib_available:
                # 使用TA-Lib计算RSI
                result[f'rsi_{period}'] = ta.RSI(result['close'].values, timeperiod=period)
            else:
                # 内部实现RSI计算
                delta = result['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # 对于前period+1个点的特殊处理
                for i in range(1, period+1):
                    avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
                    avg_loss.iloc[period] = loss.iloc[1:period+1].mean()
                
                # 处理接下来的点
                for i in range(period+1, len(delta)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
                    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
                
                # 计算RS和RSI
                rs = avg_gain / avg_loss
                result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    except Exception as e:
        logger.error(f"计算RSI指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    return result

def calculate_bollinger_bands(data, period=20, std_devs=[2, 3], use_talib=True):
    """
    计算布林带指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    period (int): 布林带周期
    std_devs (list): 标准差倍数列表
    use_talib (bool): 是否使用TA-Lib库计算
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        for std in std_devs:
            if use_talib and _talib_available:
                # 使用TA-Lib计算布林带
                upper, middle, lower = ta.BBANDS(
                    result['close'].values, timeperiod=period, 
                    nbdevup=std, nbdevdn=std, matype=0
                )
                
                suffix = '' if std == 2 else f'_{std}std'
                result[f'bb_upper{suffix}'] = upper
                result[f'bb_middle{suffix}'] = middle
                result[f'bb_lower{suffix}'] = lower
            else:
                # 内部实现布林带计算
                sma = result['close'].rolling(window=period).mean()
                rolling_std = result['close'].rolling(window=period).std()
                
                suffix = '' if std == 2 else f'_{std}std'
                result[f'bb_middle{suffix}'] = sma
                result[f'bb_upper{suffix}'] = sma + (rolling_std * std)
                result[f'bb_lower{suffix}'] = sma - (rolling_std * std)
            
            # 计算布林带宽度 (作为波动率指标)
            suffix = '' if std == 2 else f'_{std}std'
            result[f'bb_width{suffix}'] = (result[f'bb_upper{suffix}'] - result[f'bb_lower{suffix}']) / result[f'bb_middle{suffix}'] * 100
    
    except Exception as e:
        logger.error(f"计算布林带指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    return result

def calculate_moving_averages(data, periods=[10, 20, 50, 200], types=['sma', 'ema'], use_talib=True):
    """
    计算移动平均线指标
    
    参数:
    data (pandas.DataFrame): 包含'close'列的DataFrame
    periods (list): 需要计算的周期列表
    types (list): 移动平均类型 ('sma', 'ema', 'wma')
    use_talib (bool): 是否使用TA-Lib库计算
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        for period in periods:
            if 'sma' in types:
                if use_talib and _talib_available:
                    result[f'sma_{period}'] = ta.SMA(result['close'].values, timeperiod=period)
                else:
                    result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
            
            if 'ema' in types:
                if use_talib and _talib_available:
                    result[f'ema_{period}'] = ta.EMA(result['close'].values, timeperiod=period)
                else:
                    result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
            
            if 'wma' in types and use_talib and _talib_available:
                result[f'wma_{period}'] = ta.WMA(result['close'].values, timeperiod=period)
            elif 'wma' in types:
                # 内部实现加权移动平均线
                weights = np.arange(1, period + 1)
                result[f'wma_{period}'] = result['close'].rolling(period).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
    
    except Exception as e:
        logger.error(f"计算移动平均线指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    return result

def calculate_volume_indicators(data, periods=[10, 20], use_talib=True):
    """
    计算成交量相关指标
    
    参数:
    data (pandas.DataFrame): 包含'close'和'volume'列的DataFrame
    periods (list): 需要计算的周期列表
    use_talib (bool): 是否使用TA-Lib库计算
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        # 成交量移动平均
        for period in periods:
            if use_talib and _talib_available:
                result[f'volume_sma_{period}'] = ta.SMA(result['volume'].values, timeperiod=period)
            else:
                result[f'volume_sma_{period}'] = result['volume'].rolling(window=period).mean()
        
        # 计算相对成交量 (当日成交量/20日平均成交量)
        result['relative_volume'] = result['volume'] / result['volume_sma_20']
        
        # 成交量波动率 (20日成交量标准差/20日平均成交量)
        volumes = result['volume'].rolling(window=20).std()
        result['volume_volatility'] = volumes / result['volume_sma_20']
        
        # OBV (累积能量潮)
        if use_talib and _talib_available:
            result['obv'] = ta.OBV(result['close'].values, result['volume'].values)
        else:
            # 内部实现OBV计算
            obv = np.zeros(len(result))
            for i in range(1, len(result)):
                if result['close'].iloc[i] > result['close'].iloc[i-1]:
                    obv[i] = obv[i-1] + result['volume'].iloc[i]
                elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                    obv[i] = obv[i-1] - result['volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            result['obv'] = obv
    
    except Exception as e:
        logger.error(f"计算成交量指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    return result

def calculate_volatility_indicators(data, use_talib=True):
    """
    计算波动率相关指标
    
    参数:
    data (pandas.DataFrame): 包含'high', 'low', 'close'列的DataFrame
    use_talib (bool): 是否使用TA-Lib库计算
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        # ATR - 平均真实波幅
        if use_talib and _talib_available:
            result['atr'] = ta.ATR(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
        else:
            # 内部实现ATR计算
            high = result['high']
            low = result['low']
            close = result['close']
            
            # 计算真实波幅 (TR)
            tr1 = high - low  # 当日振幅
            tr2 = abs(high - close.shift())  # 当日最高与昨收差值
            tr3 = abs(low - close.shift())  # 当日最低与昨收差值
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # 计算ATR
            result['atr'] = tr.rolling(window=14).mean()
        
        # 相对ATR (ATR/收盘价)
        result['relative_atr'] = result['atr'] / result['close'] * 100
        
        # ATR变化率
        result['atr_change'] = result['atr'].pct_change(10) * 100
    
    except Exception as e:
        logger.error(f"计算波动率指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
    return result

def calculate_momentum_indicators(data, use_talib=True):
    """
    计算动量相关指标
    
    参数:
    data (pandas.DataFrame): 包含'high', 'low', 'close'列的DataFrame
    use_talib (bool): 是否使用TA-Lib库计算
    
    返回:
    pandas.DataFrame: 包含计算结果的原始DataFrame
    """
    result = data.copy()
    
    try:
        # 随机指标
        if use_talib and _talib_available:
            result['slowk'], result['slowd'] = ta.STOCH(
                result['high'].values, result['low'].values, result['close'].values,
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )
        else:
            # 内部实现随机指标计算
            high_14 = result['high'].rolling(window=14).max()
            low_14 = result['low'].rolling(window=14).min()
            
            # %K = (当前收盘价 - 14日最低价) / (14日最高价 - 14日最低价) * 100
            k_fast = 100 * (result['close'] - low_14) / (high_14 - low_14)
            
            # 计算慢速%K和%D (3日平均)
            result['slowk'] = k_fast.rolling(window=3).mean()
            result['slowd'] = result['slowk'].rolling(window=3).mean()
        
        # Williams %R
        if use_talib and _talib_available:
            result['willr'] = ta.WILLR(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
        else:
            # 内部实现威廉指标计算
            high_14 = result['high'].rolling(window=14).max()
            low_14 = result['low'].rolling(window=14).min()
            
            # %R = (14日最高价 - 当前收盘价) / (14日最高价 - 14日最低价) * -100
            result['willr'] = -100 * (high_14 - result['close']) / (high_14 - low_14)
        
        # CCI - 商品通道指数
        if use_talib and _talib_available:
            result['cci'] = ta.CCI(result['high'].values, result['low'].values, result['close'].values, timeperiod=20)
        else:
            # 内部实现CCI计算
            tp = (result['high'] + result['low'] + result['close']) / 3  # 典型价格
            tp_sma = tp.rolling(window=20).mean()  # 典型价格的20日SMA
            tp_md = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))  # 偏差的平均值
            
            # CCI = (典型价格 - 典型价格的20日SMA) / (0.015 * 偏差的平均值)
            result['cci'] = (tp - tp_sma) / (0.015 * tp_md)
        
        # MACD
        if use_talib and _talib_available:
            result['macd'], result['macd_signal'], result['macd_hist'] = ta.MACD(
                result['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            # 内部实现MACD计算
            ema_12 = result['close'].ewm(span=12, adjust=False).mean()  # 12日EMA
            ema_26 = result['close'].ewm(span=26, adjust=False).mean()  # 26日EMA
            
            # MACD线 = 12日EMA - 26日EMA
            result['macd'] = ema_12 - ema_26
            
            # 信号线 = MACD的9日EMA
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            
            # MACD柱状图 = MACD线 - 信号线
            result['macd_hist'] = result['macd'] - result['macd_signal']
    
    except Exception as e:
        logger.error(f"计算动量指标时出错: {str(e)}")
        traceback.print_exc()
        raise
    
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
    try:
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
    
    except Exception as e:
        logger.error(f"识别支撑位时出错: {str(e)}")
        traceback.print_exc()
        return [], {}

def identify_candlestick_patterns(data, use_talib=True):
    """
    识别常见蜡烛图形态
    
    参数:
    data (pandas.DataFrame): 包含'open', 'high', 'low', 'close'列的DataFrame
    use_talib (bool): 是否使用TA-Lib库识别蜡烛图形态
    
    返回:
    pandas.DataFrame: 包含识别结果的原始DataFrame
    """
    result = data.copy()
    
    try:
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
        
        # 如果可用TA-Lib，添加更多蜡烛图形态识别
        if use_talib and _talib_available:
            # 常见看涨形态
            result['cdl_hammer'] = ta.CDLHAMMER(
                result['open'].values, result['high'].values, 
                result['low'].values, result['close'].values
            )
            
            result['cdl_morning_star'] = ta.CDLMORNINGSTAR(
                result['open'].values, result['high'].values, 
                result['low'].values, result['close'].values
            )
            
            result['cdl_piercing'] = ta.CDLPIERCING(
                result['open'].values, result['high'].values, 
                result['low'].values, result['close'].values
            )
            
            result['cdl_bullish_engulfing'] = ta.CDLENGULFING(
                result['open'].values, result['high'].values, 
                result['low'].values, result['close'].values
            )
            
            # 汇总常见看涨形态
            result['bullish_pattern'] = (
                (result['cdl_hammer'] > 0) | 
                (result['cdl_morning_star'] > 0) | 
                (result['cdl_piercing'] > 0) | 
                (result['cdl_bullish_engulfing'] > 0)
            )
    
    except Exception as e:
        logger.error(f"识别蜡烛图形态时出错: {str(e)}")
        traceback.print_exc()
    
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
    
    try:
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
                if 'rsi_14' in window.columns:
                    current_rsi = window['rsi_14'].iloc[-1]
                    prev_low_rsi = window['rsi_14'].iloc[:-1].min()
                    if current_rsi > prev_low_rsi:
                        result.loc[result.index[i], 'rsi_bullish_divergence'] = True
                
                # MACD底背离
                if 'macd_hist' in window.columns:
                    current_macd_hist = window['macd_hist'].iloc[-1]
                    prev_low_macd_hist = window['macd_hist'].iloc[:-1].min()
                    if current_macd_hist > prev_low_macd_hist:
                        result.loc[result.index[i], 'macd_bullish_divergence'] = True
                
                # OBV底背离
                if 'obv' in window.columns:
                    current_obv = window['obv'].iloc[-1]
                    prev_low_obv = window['obv'].iloc[:-1].min()
                    if current_obv > prev_low_obv:
                        result.loc[result.index[i], 'obv_bullish_divergence'] = True
        
        return result
    
    except Exception as e:
        logger.error(f"检测背离过程中出错: {str(e)}")
        traceback.print_exc()
        return result