"""
数据预处理模块 - 提供数据清洗和预处理功能
"""

import pandas as pd
import numpy as np

def clean_price_data(data):
    """
    清洗价格数据
    
    参数:
    data (pandas.DataFrame): 原始数据
    
    返回:
    pandas.DataFrame: 清洗后的数据
    """
    # 复制数据，避免修改原始数据
    cleaned_data = data.copy()
    
    # 检查日期列
    if 'date' in cleaned_data.columns:
        # 确保日期格式正确
        cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
        # 按日期排序
        cleaned_data = cleaned_data.sort_values('date')
    
    # 检查必要的列是否存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in cleaned_data.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 处理缺失值
    for col in required_columns:
        # 检查是否有缺失值
        if cleaned_data[col].isnull().any():
            # 用前一个值填充
            cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
            # 如果第一个值是缺失的，用后一个值填充
            cleaned_data[col] = cleaned_data[col].fillna(method='bfill')
    
    # 确保价格数据合理
    # 确保high >= low
    invalid_hl = cleaned_data['high'] < cleaned_data['low']
    if invalid_hl.any():
        # 交换不合理的high和low
        temp = cleaned_data.loc[invalid_hl, 'high'].copy()
        cleaned_data.loc[invalid_hl, 'high'] = cleaned_data.loc[invalid_hl, 'low']
        cleaned_data.loc[invalid_hl, 'low'] = temp
    
    # 确保high >= max(open, close)
    cleaned_data['max_oc'] = cleaned_data[['open', 'close']].max(axis=1)
    invalid_h = cleaned_data['high'] < cleaned_data['max_oc']
    if invalid_h.any():
        cleaned_data.loc[invalid_h, 'high'] = cleaned_data.loc[invalid_h, 'max_oc']
    
    # 确保low <= min(open, close)
    cleaned_data['min_oc'] = cleaned_data[['open', 'close']].min(axis=1)
    invalid_l = cleaned_data['low'] > cleaned_data['min_oc']
    if invalid_l.any():
        cleaned_data.loc[invalid_l, 'low'] = cleaned_data.loc[invalid_l, 'min_oc']
    
    # 删除临时列
    cleaned_data = cleaned_data.drop(['max_oc', 'min_oc'], axis=1)
    
    # 确保成交量为正
    cleaned_data.loc[cleaned_data['volume'] < 0, 'volume'] = 0
    
    return cleaned_data

def normalize_data(data, columns=None):
    """
    标准化数据
    
    参数:
    data (pandas.DataFrame): 原始数据
    columns (list): 要标准化的列名列表，默认为None(标准化所有数值列)
    
    返回:
    pandas.DataFrame: 标准化后的数据
    """
    from sklearn.preprocessing import StandardScaler
    
    # 复制数据，避免修改原始数据
    normalized_data = data.copy()
    
    # 如果未指定列，则选择所有数值列
    if columns is None:
        columns = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建标准化器
    scaler = StandardScaler()
    
    # 标准化数据
    normalized_data[columns] = scaler.fit_transform(normalized_data[columns])
    
    return normalized_data

def split_train_test(data, test_size=0.2, by_date=True):
    """
    将数据分割为训练集和测试集
    
    参数:
    data (pandas.DataFrame): 原始数据
    test_size (float): 测试集比例，默认为0.2
    by_date (bool): 是否按日期分割，默认为True
    
    返回:
    tuple: (训练集, 测试集)
    """
    if by_date and 'date' in data.columns:
        # 按日期排序
        data = data.sort_values('date')
        
        # 计算分割点
        split_idx = int(len(data) * (1 - test_size))
        
        # 分割数据
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
    else:
        # 随机分割
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    return train, test

def resample_data(data, freq='W'):
    """
    重采样数据到不同的频率
    
    参数:
    data (pandas.DataFrame): 原始数据，需要包含'date'列
    freq (str): 重采样频率，默认为'W'(周)
                可选: 'D'(日), 'W'(周), 'M'(月), 'Q'(季), 'Y'(年)
    
    返回:
    pandas.DataFrame: 重采样后的数据
    """
    if 'date' not in data.columns:
        raise ValueError("数据必须包含'date'列")
    
    # 设置日期为索引
    data_indexed = data.set_index('date')
    
    # 重采样规则
    resampled = pd.DataFrame()
    resampled['open'] = data_indexed['open'].resample(freq).first()
    resampled['high'] = data_indexed['high'].resample(freq).max()
    resampled['low'] = data_indexed['low'].resample(freq).min()
    resampled['close'] = data_indexed['close'].resample(freq).last()
    resampled['volume'] = data_indexed['volume'].resample(freq).sum()
    
    # 重置索引
    resampled = resampled.reset_index()
    
    return resampled