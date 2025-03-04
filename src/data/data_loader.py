"""
数据加载模块 - 提供从不同来源加载价格数据的功能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_from_yahoo(symbol, period="1y", interval="1d", retry_count=3):
    """
    从Yahoo Finance API加载股票数据
    
    参数:
    symbol (str): 股票代码，例如 'AAPL'
    period (str): 时间跨度，可选值: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    interval (str): 时间间隔，可选值: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    retry_count (int): 重试次数
    
    返回:
    pandas.DataFrame: 包含日期、开盘价、最高价、最低价、收盘价和成交量的DataFrame
    """
    try:
        # 尝试导入yfinance库
        import yfinance as yf
        
        logger.info(f"从Yahoo Finance加载 {symbol} 数据，周期: {period}，间隔: {interval}")
        
        # 使用重试机制，因为Yahoo API可能不稳定
        for attempt in range(retry_count):
            try:
                # 下载数据
                data = yf.download(
                    tickers=symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False
                )
                
                # 如果数据为空，抛出异常
                if data.empty:
                    raise ValueError(f"未找到 {symbol} 的数据")
                
                # 处理索引和列名
                data = data.reset_index()
                data.columns = [col.lower() for col in data.columns]
                
                # 确保日期列名一致
                if 'date' not in data.columns and 'datetime' in data.columns:
                    data = data.rename(columns={'datetime': 'date'})
                
                logger.info(f"成功加载 {len(data)} 条数据记录")
                return data
                
            except Exception as e:
                logger.warning(f"尝试 {attempt+1}/{retry_count} 失败: {str(e)}")
                if attempt == retry_count - 1:
                    logger.error(f"从Yahoo Finance加载数据失败: {str(e)}")
                    raise
                
    except ImportError:
        logger.error("未安装yfinance库。请使用pip install yfinance安装。")
        raise ImportError("需要yfinance库。请使用pip install yfinance安装。")

def load_from_csv(filepath, date_column='date', open_column='open', high_column='high', 
                 low_column='low', close_column='close', volume_column='volume',
                 date_format='%Y-%m-%d'):
    """
    从CSV文件加载股票数据
    
    参数:
    filepath (str): CSV文件路径
    date_column (str): 日期列名
    open_column (str): 开盘价列名
    high_column (str): 最高价列名
    low_column (str): 最低价列名
    close_column (str): 收盘价列名
    volume_column (str): 成交量列名
    date_format (str): 日期格式字符串
    
    返回:
    pandas.DataFrame: 包含日期、开盘价、最高价、最低价、收盘价和成交量的DataFrame
    """
    try:
        logger.info(f"从CSV文件加载数据: {filepath}")
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 加载CSV文件
        data = pd.read_csv(filepath)
        
        # 检查必要列是否存在
        required_columns = [date_column, close_column]
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"CSV文件缺少必要列: {col}")
                raise ValueError(f"CSV文件缺少必要列: {col}")
        
        # 处理日期列
        try:
            data[date_column] = pd.to_datetime(data[date_column], format=date_format)
        except Exception as e:
            logger.error(f"日期格式转换失败: {str(e)}")
            raise ValueError(f"日期格式转换失败。请确保日期列格式为 {date_format}，或指定正确的格式。")
        
        # 重命名列以匹配标准格式
        column_mapping = {
            date_column: 'date',
            open_column: 'open',
            high_column: 'high',
            low_column: 'low',
            close_column: 'close',
            volume_column: 'volume'
        }
        
        # 只重命名存在的列
        column_mapping = {k: v for k, v in column_mapping.items() if k in data.columns}
        data = data.rename(columns=column_mapping)
        
        # 添加缺失的列（除了日期和收盘价）
        if 'open' not in data.columns:
            data['open'] = data['close']
        if 'high' not in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns:
            data['low'] = data['close']
        if 'volume' not in data.columns:
            data['volume'] = 0
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype(int)
        
        # 按日期排序
        data = data.sort_values('date')
        
        logger.info(f"成功从CSV加载 {len(data)} 条数据记录")
        return data
        
    except Exception as e:
        logger.error(f"从CSV加载数据失败: {str(e)}")
        raise

def load_from_custom_api(api_url, params=None, headers=None, date_field='date', price_fields=None):
    """
    从自定义API加载股票数据
    
    参数:
    api_url (str): API端点URL
    params (dict): API请求参数
    headers (dict): API请求头
    date_field (str): 响应中的日期字段名
    price_fields (dict): 字段映射，格式为{'open': 'openPrice', 'high': 'highPrice', ...}
    
    返回:
    pandas.DataFrame: 包含日期、开盘价、最高价、最低价、收盘价和成交量的DataFrame
    """
    try:
        logger.info(f"从自定义API加载数据: {api_url}")
        
        # 默认请求头
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        # 默认字段映射
        if price_fields is None:
            price_fields = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # 发送请求
        response = requests.get(api_url, params=params, headers=headers)
        
        # 检查响应状态
        if response.status_code != 200:
            logger.error(f"API请求失败，状态码: {response.status_code}")
            raise ValueError(f"API请求失败，状态码: {response.status_code}，响应: {response.text}")
        
        # 解析响应
        try:
            json_data = response.json()
        except Exception as e:
            logger.error(f"解析API响应JSON失败: {str(e)}")
            raise ValueError(f"解析API响应失败: {str(e)}")
        
        # 提取数据
        if isinstance(json_data, list):
            # 直接是数据列表
            items = json_data
        else:
            # 可能嵌套在某个字段中，尝试常见的字段名
            for field in ['data', 'items', 'results', 'quotes', 'candles', 'ohlcv']:
                if field in json_data and isinstance(json_data[field], list):
                    items = json_data[field]
                    break
            else:
                logger.error("无法从API响应中找到数据数组")
                raise ValueError("无法从API响应中找到数据数组")
        
        # 转换为DataFrame
        data = pd.DataFrame(items)
        
        # 检查必要字段是否存在
        if date_field not in data.columns:
            logger.error(f"API响应缺少日期字段: {date_field}")
            raise ValueError(f"API响应缺少日期字段: {date_field}")
        
        if price_fields['close'] not in data.columns:
            logger.error(f"API响应缺少收盘价字段: {price_fields['close']}")
            raise ValueError(f"API响应缺少收盘价字段: {price_fields['close']}")
        
        # 重命名字段
        data = data.rename(columns={
            date_field: 'date',
            price_fields['open']: 'open' if price_fields['open'] in data.columns else None,
            price_fields['high']: 'high' if price_fields['high'] in data.columns else None,
            price_fields['low']: 'low' if price_fields['low'] in data.columns else None,
            price_fields['close']: 'close',
            price_fields['volume']: 'volume' if price_fields['volume'] in data.columns else None
        })
        
        # 移除None映射
        data = data.rename(columns={k: v for k, v in data.columns.items() if v is not None})
        
        # 转换日期
        data['date'] = pd.to_datetime(data['date'])
        
        # 添加缺失的列
        if 'open' not in data.columns:
            data['open'] = data['close']
        if 'high' not in data.columns:
            data['high'] = data['close']
        if 'low' not in data.columns:
            data['low'] = data['close']
        if 'volume' not in data.columns:
            data['volume'] = 0
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype(int)
        
        # 按日期排序
        data = data.sort_values('date')
        
        logger.info(f"成功从API加载 {len(data)} 条数据记录")
        return data
        
    except Exception as e:
        logger.error(f"从自定义API加载数据失败: {str(e)}")
        raise

def generate_sample_data(days=252, start_date=None, start_price=100, volatility=0.01, trend=0.0001, seed=None):
    """
    生成模拟的价格数据用于测试
    
    参数:
    days (int): 生成的数据天数
    start_date (datetime): 起始日期，默认为今天向前推days天
    start_price (float): 起始价格
    volatility (float): 每日波动率
    trend (float): 每日价格趋势因子
    seed (int): 随机数种子，用于复现结果
    
    返回:
    pandas.DataFrame: 包含模拟价格数据的DataFrame
    """
    logger.info(f"生成 {days} 天的模拟价格数据")
    
    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
    
    # 设置起始日期
    if start_date is None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
    else:
        end_date = start_date + timedelta(days=days)
    
    # 生成交易日序列（排除周末）
    all_dates = []
    current_date = start_date
    while len(all_dates) < days and current_date <= end_date:
        # 周一至周五（0=周一，6=周日）
        if current_date.weekday() < 5:
            all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # 生成价格数据
    closes = [start_price]
    opens = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(1, days):
        # 添加随机波动和趋势
        daily_return = np.random.normal(trend, volatility)
        close_price = closes[-1] * (1 + daily_return)
        closes.append(close_price)
        
        # 生成日内价格
        daily_volatility = volatility * closes[-1]
        open_price = close_price * (1 + np.random.normal(0, 0.3) * daily_return)
        high_price = max(close_price, open_price) + abs(np.random.normal(0, 1.5) * daily_volatility)
        low_price = min(close_price, open_price) - abs(np.random.normal(0, 1.5) * daily_volatility)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        
        # 生成成交量（与价格波动相关）
        base_volume = 1000000  # 基础成交量
        volume_factor = 1 + 5 * abs(daily_return) / volatility  # 价格波动越大，成交量越大
        volume = int(base_volume * volume_factor * (1 + np.random.normal(0, 0.3)))
        volumes.append(max(1000, volume))  # 确保成交量为正
    
    # 添加首日数据
    opens.insert(0, start_price * (1 - 0.001))
    highs.insert(0, start_price * (1 + 0.005))
    lows.insert(0, start_price * (1 - 0.005))
    volumes.insert(0, 1000000)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': all_dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    logger.info(f"成功生成 {len(data)} 条模拟数据记录")
    return data