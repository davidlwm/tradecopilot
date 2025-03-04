"""
数据加载模块 - 提供从各种来源加载数据的功能
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def load_from_csv(file_path):
    """
    从CSV文件加载数据
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    pandas.DataFrame: 加载的数据
    """
    data = pd.read_csv(file_path)
    
    # 尝试将日期列转换为datetime
    date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            data[col] = pd.to_datetime(data[col])
        except:
            pass
    
    return data

def load_from_yahoo(ticker, period='1y', interval='1d'):
    """
    从Yahoo Finance加载数据
    
    参数:
    ticker (str): 股票代码
    period (str): 时间段 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    interval (str): 时间间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    返回:
    pandas.DataFrame: 加载的数据
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    # 重设索引，将日期作为列
    data = data.reset_index()
    
    # 重命名列名以符合我们的标准
    data = data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    return data

def load_from_custom_api(api_url, params=None, headers=None):
    """
    从自定义API加载数据
    
    参数:
    api_url (str): API URL
    params (dict): 请求参数
    headers (dict): 请求头
    
    返回:
    pandas.DataFrame: 加载的数据
    """
    import requests
    
    response = requests.get(api_url, params=params, headers=headers)
    
    if response.status_code == 200:
        # 假设API返回JSON格式
        data = pd.DataFrame(response.json())
        return data
    else:
        raise Exception(f"API请求失败: {response.status_code}")

def generate_sample_data(days=200, volatility=0.02, trend=0.0001):
    """
    生成示例数据用于测试
    
    参数:
    days (int): 生成的天数
    volatility (float): 价格波动率
    trend (float): 价格趋势系数
    
    返回:
    pandas.DataFrame: 生成的示例数据
    """
    import numpy as np
    
    # 生成日期
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    
    # 模拟价格
    np.random.seed(42)  # 固定随机种子以确保可重复性
    
    # 起始价格
    price = 100.0
    
    # 生成价格序列
    prices = [price]
    for i in range(1, len(dates)):
        # 添加波动和趋势
        change = np.random.normal(trend, volatility)
        price *= (1 + change)
        prices.append(price)
    
    # 为了模拟更真实的数据，创建OHLC数据
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # 生成开盘、最高和最低价格
    daily_volatility = volatility / 2
    df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, daily_volatility, len(df)))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, daily_volatility, len(df))))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, daily_volatility, len(df))))
    
    # 第一天没有前一天的收盘价，所以我们设置开盘价等于收盘价
    df.loc[0, 'open'] = df.loc[0, 'close']
    
    # 生成成交量
    base_volume = 1000000  # 基础成交量
    df['volume'] = base_volume * (1 + np.random.normal(0, 0.3, len(df)))
    df['volume'] = df['volume'].astype(int)
    
    # 确保OHLC数据合理
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df