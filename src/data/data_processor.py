"""
数据处理模块 - 提供数据清洗、预处理和转换功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_price_data(data, fill_missing=True, remove_duplicates=True, ensure_columns=True):
    """
    清洗价格数据，处理缺失值、重复值和异常值
    
    参数:
    data (pandas.DataFrame): 原始价格数据
    fill_missing (bool): 是否填充缺失值
    remove_duplicates (bool): 是否删除重复行
    ensure_columns (bool): 是否确保必要列存在
    
    返回:
    pandas.DataFrame: 清洗后的价格数据
    """
    try:
        logger.info("开始清洗价格数据")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 检查必要列是否存在
        required_columns = ['date', 'close']
        for col in required_columns:
            if col not in df.columns:
                msg = f"数据缺少必要列: {col}"
                logger.error(msg)
                raise ValueError(msg)
        
        # 确保日期列为datetime类型
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            logger.info("转换日期列为datetime类型")
        
        # 删除重复行
        if remove_duplicates and len(df) > 0:
            orig_len = len(df)
            df = df.drop_duplicates(subset='date', keep='first')
            if len(df) < orig_len:
                logger.info(f"删除了 {orig_len - len(df)} 行重复数据")
        
        # 按日期排序
        df = df.sort_values('date')
        logger.info("按日期对数据进行排序")
        
        # 确保所有必要列存在
        if ensure_columns:
            if 'open' not in df.columns:
                df['open'] = df['close']
                logger.info("添加缺失的open列，使用close值")
            
            if 'high' not in df.columns:
                df['high'] = df['close']
                logger.info("添加缺失的high列，使用close值")
            
            if 'low' not in df.columns:
                df['low'] = df['close']
                logger.info("添加缺失的low列，使用close值")
            
            if 'volume' not in df.columns:
                df['volume'] = 0
                logger.info("添加缺失的volume列，使用0值")
        
        # 处理缺失值
        if fill_missing:
            # 检查每列的缺失值
            for col in df.columns:
                if col == 'date':
                    continue
                
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    logger.info(f"列 {col} 有 {missing_count} 个缺失值")
                    
                    # 对于价格列，使用前向填充
                    if col in ['open', 'high', 'low', 'close']:
                        df[col] = df[col].fillna(method='ffill')
                        # 如果仍有缺失值（如第一行），使用后向填充
                        df[col] = df[col].fillna(method='bfill')
                    
                    # 对于成交量，使用0填充
                    elif col == 'volume':
                        df[col] = df[col].fillna(0)
        
        # 检查并处理异常值
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # 检查负值
                neg_count = (df[col] <= 0).sum()
                if neg_count > 0:
                    logger.warning(f"列 {col} 有 {neg_count} 个非正值，将被替换")
                    # 使用前一天的值替换
                    df.loc[df[col] <= 0, col] = np.nan
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 确保高低价合理
        if 'high' in df.columns and 'low' in df.columns:
            # 确保high >= low
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                logger.warning(f"发现 {invalid_hl} 行high值小于low值，交换这些值")
                mask = df['high'] < df['low']
                df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
        
        # 确保开收盘价在高低价范围内
        for col in ['open', 'close']:
            if col in df.columns:
                # 确保价格在high和low之间
                invalid_high = (df[col] > df['high']).sum()
                if invalid_high > 0:
                    logger.warning(f"发现 {invalid_high} 行{col}值大于high值，修正这些值")
                    df.loc[df[col] > df['high'], col] = df.loc[df[col] > df['high'], 'high']
                
                invalid_low = (df[col] < df['low']).sum()
                if invalid_low > 0:
                    logger.warning(f"发现 {invalid_low} 行{col}值小于low值，修正这些值")
                    df.loc[df[col] < df['low'], col] = df.loc[df[col] < df['low'], 'low']
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"数据清洗完成，最终数据集包含 {len(df)} 行")
        return df
    
    except Exception as e:
        logger.error(f"数据清洗过程中出错: {str(e)}")
        raise

def normalize_data(data, price_cols=['open', 'high', 'low', 'close'], method='min_max'):
    """
    标准化价格数据，用于机器学习模型
    
    参数:
    data (pandas.DataFrame): 价格数据
    price_cols (list): 需要标准化的价格列
    method (str): 标准化方法，可选值: 'min_max', 'z_score', 'log'
    
    返回:
    pandas.DataFrame: 标准化后的数据
    """
    try:
        logger.info(f"使用 {method} 方法标准化数据")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        if method == 'min_max':
            # Min-Max标准化 (缩放到0-1范围)
            for col in price_cols:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            
            logger.info("使用Min-Max方法将价格缩放到0-1范围")
            
        elif method == 'z_score':
            # Z-score标准化 (转换为均值为0，标准差为1)
            for col in price_cols:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[f'{col}_norm'] = (df[col] - mean) / std
            
            logger.info("使用Z-score方法将价格标准化")
            
        elif method == 'log':
            # 对数转换 (处理非线性关系和偏态分布)
            for col in price_cols:
                if col in df.columns:
                    # 确保值为正
                    if (df[col] <= 0).any():
                        min_positive = df[df[col] > 0][col].min()
                        df[f'{col}_norm'] = np.log(df[col].clip(lower=min_positive))
                    else:
                        df[f'{col}_norm'] = np.log(df[col])
            
            logger.info("使用对数转换方法处理价格数据")
            
        else:
            logger.error(f"未知的标准化方法: {method}")
            raise ValueError(f"未知的标准化方法: {method}。支持的方法: 'min_max', 'z_score', 'log'")
        
        return df
    
    except Exception as e:
        logger.error(f"数据标准化过程中出错: {str(e)}")
        raise

def split_train_test(data, test_size=0.2, validation_size=0.0):
    """
    将数据分割为训练集、测试集和可选的验证集
    
    参数:
    data (pandas.DataFrame): 原始数据
    test_size (float): 测试集比例 (0-1)
    validation_size (float): 验证集比例 (0-1)
    
    返回:
    tuple: 根据参数返回 (train, test) 或 (train, validation, test)
    """
    try:
        if test_size < 0 or test_size >= 1:
            raise ValueError("test_size必须在0到1之间")
        
        if validation_size < 0 or validation_size >= 1:
            raise ValueError("validation_size必须在0到1之间")
        
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size必须小于1")
        
        # 按时间顺序分割数据
        train_size = 1 - test_size - validation_size
        
        total_rows = len(data)
        train_rows = int(total_rows * train_size)
        validation_rows = int(total_rows * validation_size)
        
        # 分割数据
        train = data.iloc[:train_rows].copy()
        
        if validation_size > 0:
            validation = data.iloc[train_rows:train_rows+validation_rows].copy()
            test = data.iloc[train_rows+validation_rows:].copy()
            logger.info(f"数据分割完成: 训练集 {len(train)} 行, 验证集 {len(validation)} 行, 测试集 {len(test)} 行")
            return train, validation, test
        else:
            test = data.iloc[train_rows:].copy()
            logger.info(f"数据分割完成: 训练集 {len(train)} 行, 测试集 {len(test)} 行")
            return train, test
    
    except Exception as e:
        logger.error(f"分割数据过程中出错: {str(e)}")
        raise

def resample_data(data, timeframe='W', agg_dict=None):
    """
    将数据重采样为不同的时间周期
    
    参数:
    data (pandas.DataFrame): 原始价格数据，需要包含'date'列
    timeframe (str): pandas重采样周期，如'W'(周),'M'(月),'Q'(季度),'Y'(年)
    agg_dict (dict): 聚合方法，默认为标准OHLCV聚合
    
    返回:
    pandas.DataFrame: 重采样后的数据
    """
    try:
        logger.info(f"将数据重采样为 {timeframe} 周期")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 确保日期列为索引，以便重采样
        if 'date' in df.columns:
            df = df.set_index('date')
        
        # 默认的聚合方法
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        # 仅使用实际存在的列
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        # 执行重采样
        resampled = df.resample(timeframe).agg(agg_dict)
        
        # 重置索引
        resampled = resampled.reset_index()
        
        logger.info(f"重采样完成，新数据集包含 {len(resampled)} 行")
        return resampled
    
    except Exception as e:
        logger.error(f"重采样数据过程中出错: {str(e)}")
        raise

def add_date_features(data):
    """
    添加基于日期的特征，如星期几、月份等
    
    参数:
    data (pandas.DataFrame): 包含'date'列的价格数据
    
    返回:
    pandas.DataFrame: 增加了日期特征的数据
    """
    try:
        logger.info("添加基于日期的特征")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 确保日期列存在且为datetime类型
        if 'date' not in df.columns:
            logger.error("数据中缺少'date'列")
            raise ValueError("数据中缺少'date'列")
        
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # 添加星期几特征 (0=周一，6=周日)
        df['weekday'] = df['date'].dt.weekday
        
        # 添加月份特征 (1-12)
        df['month'] = df['date'].dt.month
        
        # 添加季度特征 (1-4)
        df['quarter'] = df['date'].dt.quarter
        
        # 添加是否月初/月末特征
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # 添加是否季初/季末特征
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # 添加是否年初/年末特征
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        logger.info("日期特征添加完成")
        return df
    
    except Exception as e:
        logger.error(f"添加日期特征过程中出错: {str(e)}")
        raise

def detect_outliers(data, cols=['open', 'high', 'low', 'close', 'volume'], method='iqr', threshold=3):
    """
    检测并标记异常值
    
    参数:
    data (pandas.DataFrame): 价格数据
    cols (list): 要检查的列
    method (str): 检测方法，'iqr'或'zscore'
    threshold (float): 判定为异常值的阈值
    
    返回:
    pandas.DataFrame: 包含异常值标记的数据
    """
    try:
        logger.info(f"使用 {method} 方法检测异常值")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 为每列创建异常值标记
        for col in cols:
            if col not in df.columns:
                logger.warning(f"列 {col} 不存在，跳过异常值检测")
                continue
            
            # 创建标记列
            outlier_col = f'{col}_outlier'
            df[outlier_col] = 0
            
            if method == 'iqr':
                # IQR方法 (四分位距法)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 标记异常值
                df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), outlier_col] = 1
                
                outlier_count = df[outlier_col].sum()
                logger.info(f"列 {col} 使用IQR方法检测到 {outlier_count} 个异常值")
                
            elif method == 'zscore':
                # Z-score方法
                mean = df[col].mean()
                std = df[col].std()
                
                # 标记异常值
                df.loc[abs(df[col] - mean) > threshold * std, outlier_col] = 1
                
                outlier_count = df[outlier_col].sum()
                logger.info(f"列 {col} 使用Z-score方法检测到 {outlier_count} 个异常值")
                
            else:
                logger.error(f"未知的异常值检测方法: {method}")
                raise ValueError(f"未知的异常值检测方法: {method}。支持的方法: 'iqr', 'zscore'")
        
        return df
    
    except Exception as e:
        logger.error(f"检测异常值过程中出错: {str(e)}")
        raise

def impute_missing_prices(data, method='linear'):
    """
    填充缺失的价格数据，主要用于处理非交易日
    
    参数:
    data (pandas.DataFrame): 价格数据，需要包含'date'列
    method (str): 填充方法，'linear', 'ffill', 'bfill'或'spline'
    
    返回:
    pandas.DataFrame: 填充后的数据
    """
    try:
        logger.info(f"使用 {method} 方法填充缺失的价格数据")
        
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 确保日期列存在
        if 'date' not in df.columns:
            logger.error("数据中缺少'date'列")
            raise ValueError("数据中缺少'date'列")
        
        # 确保日期是datetime类型
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # 设置日期为索引
        df = df.set_index('date')
        
        # 创建完整的日期范围
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        
        # 重新索引数据，添加缺失的日期
        df = df.reindex(date_range)
        
        # 填充缺失值
        price_cols = ['open', 'high', 'low', 'close']
        if method == 'linear':
            # 线性插值
            df[price_cols] = df[price_cols].interpolate(method='linear')
            logger.info("使用线性插值填充缺失价格")
            
        elif method == 'ffill':
            # 前向填充
            df[price_cols] = df[price_cols].fillna(method='ffill')
            logger.info("使用前向填充法填充缺失价格")
            
        elif method == 'bfill':
            # 后向填充
            df[price_cols] = df[price_cols].fillna(method='bfill')
            logger.info("使用后向填充法填充缺失价格")
            
        elif method == 'spline':
            # 样条插值
            df[price_cols] = df[price_cols].interpolate(method='spline', order=3)
            logger.info("使用样条插值填充缺失价格")
            
        else:
            logger.error(f"未知的填充方法: {method}")
            raise ValueError(f"未知的填充方法: {method}。支持的方法: 'linear', 'ffill', 'bfill', 'spline'")
        
        # 填充成交量，使用0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # 重置索引
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        
        # 去除周末
        if not df.empty:
            df = df[~((df['date'].dt.weekday == 5) | (df['date'].dt.weekday == 6))]
        
        logger.info(f"缺失价格填充完成，最终数据集包含 {len(df)} 行")
        return df
    
    except Exception as e:
        logger.error(f"填充缺失价格过程中出错: {str(e)}")
        raise