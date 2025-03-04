"""
数据加载和处理模块的单元测试
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import generate_sample_data
from src.data.data_processor import (
    clean_price_data, normalize_data, split_train_test, 
    resample_data, add_trading_days, add_date_features, remove_outliers
)

class TestDataProcessing(unittest.TestCase):
    """测试数据处理模块的功能"""
    
    def setUp(self):
        """测试前准备工作"""
        # 生成测试用的样本数据
        self.sample_data = generate_sample_data(days=100, volatility=0.02, trend=0.0001, seed=42)
        
        # 故意添加一些问题数据用于测试清洗功能
        # 创建一个带有问题的副本
        self.dirty_data = self.sample_data.copy()
        
        # 添加一些缺失值
        self.dirty_data.loc[10, 'close'] = np.nan
        self.dirty_data.loc[20, 'open'] = np.nan
        self.dirty_data.loc[30, 'high'] = np.nan
        self.dirty_data.loc[40, 'low'] = np.nan
        self.dirty_data.loc[50, 'volume'] = np.nan
        
        # 添加一些异常值
        self.dirty_data.loc[15, 'close'] = -10.0  # 负价格
        self.dirty_data.loc[25, 'high'] = 1000000.0  # 极高价格
        self.dirty_data.loc[35, 'volume'] = -5000  # 负成交量
        
        # 添加一些矛盾数据
        self.dirty_data.loc[45, 'high'] = self.dirty_data.loc[45, 'low'] * 0.9  # high < low
        self.dirty_data.loc[55, 'close'] = self.dirty_data.loc[55, 'high'] * 1.1  # close > high
        self.dirty_data.loc[65, 'open'] = self.dirty_data.loc[65, 'low'] * 0.9  # open < low
        
        # 添加重复行
        duplicate_row = self.dirty_data.iloc[5].copy()
        self.dirty_data = pd.concat([self.dirty_data, pd.DataFrame([duplicate_row])], ignore_index=True)
    
    def test_clean_price_data(self):
        """测试数据清洗功能"""
        # 清洗脏数据
        cleaned_data = clean_price_data(self.dirty_data)
        
        # 验证没有缺失值
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)
        
        # 验证没有负的价格值
        for col in ['open', 'high', 'low', 'close']:
            self.assertTrue(all(cleaned_data[col] > 0))
        
        # 验证没有负的成交量
        self.assertTrue(all(cleaned_data['volume'] >= 0))
        
        # 验证high >= low的规则
        self.assertTrue(all(cleaned_data['high'] >= cleaned_data['low']))
        
        # 验证high >= close和high >= open的规则
        self.assertTrue(all(cleaned_data['high'] >= cleaned_data['close']))
        self.assertTrue(all(cleaned_data['high'] >= cleaned_data['open']))
        
        # 验证low <= close和low <= open的规则
        self.assertTrue(all(cleaned_data['low'] <= cleaned_data['close']))
        self.assertTrue(all(cleaned_data['low'] <= cleaned_data['open']))
        
        # 验证没有重复日期
        self.assertEqual(len(cleaned_data['date'].unique()), len(cleaned_data))
        
        # 验证数据是按日期排序的
        self.assertTrue(cleaned_data['date'].is_monotonic_increasing)
    
    def test_normalize_data(self):
        """测试数据标准化功能"""
        # 测试Min-Max标准化
        norm_minmax = normalize_data(self.sample_data, method='min_max')
        
        # 验证是否添加了标准化后的列
        self.assertIn('open_norm', norm_minmax.columns)
        self.assertIn('high_norm', norm_minmax.columns)
        self.assertIn('low_norm', norm_minmax.columns)
        self.assertIn('close_norm', norm_minmax.columns)
        
        # 验证标准化值在0-1范围内
        for col in ['open_norm', 'high_norm', 'low_norm', 'close_norm']:
            self.assertTrue(all(0 <= norm_minmax[col]) and all(norm_minmax[col] <= 1))
        
        # 测试Z-score标准化
        norm_zscore = normalize_data(self.sample_data, method='z_score')
        
        # 验证是否添加了标准化后的列
        self.assertIn('open_norm', norm_zscore.columns)
        self.assertIn('close_norm', norm_zscore.columns)
        
        # 验证Z-score标准化后的均值接近0，标准差接近1
        for col in ['open_norm', 'high_norm', 'low_norm', 'close_norm']:
            self.assertAlmostEqual(norm_zscore[col].mean(), 0, places=1)
            self.assertAlmostEqual(norm_zscore[col].std(), 1, places=1)
        
        # 测试对数转换
        norm_log = normalize_data(self.sample_data, method='log')
        
        # 验证是否添加了转换后的列
        self.assertIn('open_norm', norm_log.columns)
        self.assertIn('close_norm', norm_log.columns)
        
        # 验证对数转换后的值大于0
        for col in ['open_norm', 'high_norm', 'low_norm', 'close_norm']:
            self.assertTrue(all(norm_log[col] > 0))
    
    def test_split_train_test(self):
        """测试数据分割功能"""
        # 测试简单的训练/测试分割
        train, test = split_train_test(self.sample_data, test_size=0.2)
        
        # 验证分割比例
        self.assertAlmostEqual(len(train) / len(self.sample_data), 0.8, places=1)
        self.assertAlmostEqual(len(test) / len(self.sample_data), 0.2, places=1)
        
        # 验证总行数一致
        self.assertEqual(len(train) + len(test), len(self.sample_data))
        
        # 验证时间顺序 - 训练集日期都在测试集之前
        self.assertTrue(train['date'].max() < test['date'].min())
        
        # 测试带验证集的分割
        train, val, test = split_train_test(self.sample_data, test_size=0.2, validation_size=0.1)
        
        # 验证分割比例
        self.assertAlmostEqual(len(train) / len(self.sample_data), 0.7, places=1)
        self.assertAlmostEqual(len(val) / len(self.sample_data), 0.1, places=1)
        self.assertAlmostEqual(len(test) / len(self.sample_data), 0.2, places=1)
        
        # 验证总行数一致
        self.assertEqual(len(train) + len(val) + len(test), len(self.sample_data))
        
        # 验证时间顺序
        self.assertTrue(train['date'].max() < val['date'].min())
        self.assertTrue(val['date'].max() < test['date'].min())
    
    def test_resample_data(self):
        """测试数据重采样功能"""
        # 测试按周重采样
        weekly_data = resample_data(self.sample_data, timeframe='W')
        
        # 验证行数减少
        self.assertLess(len(weekly_data), len(self.sample_data))
        
        # 验证必要的列存在
        for col in ['date', 'open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, weekly_data.columns)
        
        # 测试按月重采样
        monthly_data = resample_data(self.sample_data, timeframe='M')
        
        # 验证行数减少
        self.assertLess(len(monthly_data), len(weekly_data))
        
        # 验证月度最高价高于或等于任何单日的高价
        for i in range(len(monthly_data)):
            month_start = monthly_data.iloc[i]['date'].replace(day=1)
            month_end = (month_start + pd.DateOffset(months=1)).replace(day=1) - pd.DateOffset(days=1)
            
            daily_data_in_month = self.sample_data[(self.sample_data['date'] >= month_start) & 
                                                  (self.sample_data['date'] <= month_end)]
            
            if not daily_data_in_month.empty:
                # 验证月最高价高于或等于该月任一日的最高价
                self.assertGreaterEqual(monthly_data.iloc[i]['high'], daily_data_in_month['high'].max())
                
                # 验证月最低价低于或等于该月任一日的最低价
                self.assertLessEqual(monthly_data.iloc[i]['low'], daily_data_in_month['low'].min())
                
                # 验证月成交量是该月所有天成交量之和
                self.assertAlmostEqual(monthly_data.iloc[i]['volume'], daily_data_in_month['volume'].sum())
    
    def test_add_trading_days(self):
        """测试添加交易日序号功能"""
        # 添加交易日序号
        data_with_days = add_trading_days(self.sample_data)
        
        # 验证是否添加了trading_day列
        self.assertIn('trading_day', data_with_days.columns)
        
        # 验证交易日序号从1开始递增
        self.assertEqual(data_with_days['trading_day'].iloc[0], 1)
        self.assertEqual(data_with_days['trading_day'].iloc[-1], len(data_with_days))
        
        # 验证序号是连续的
        self.assertTrue(all(data_with_days['trading_day'].diff().iloc[1:] == 1))
    
    def test_add_date_features(self):
        """测试添加日期特征功能"""
        # 添加日期特征
        data_with_features = add_date_features(self.sample_data)
        
        # 验证是否添加了日期特征列
        date_features = ['year', 'month', 'day', 'weekday', 'quarter', 
                         'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                         'is_month_start', 'is_month_end', 'is_quarter_end']
        
        for feature in date_features:
            self.assertIn(feature, data_with_features.columns)
        
        # 验证特征值是否正确
        for i in range(len(data_with_features)):
            date = data_with_features['date'].iloc[i]
            
            self.assertEqual(data_with_features['year'].iloc[i], date.year)
            self.assertEqual(data_with_features['month'].iloc[i], date.month)
            self.assertEqual(data_with_features['day'].iloc[i], date.day)
            self.assertEqual(data_with_features['weekday'].iloc[i], date.weekday())
            self.assertEqual(data_with_features['quarter'].iloc[i], (date.month - 1) // 3 + 1)
            
            # 验证周期性特征在正确的范围内
            self.assertTrue(-1 <= data_with_features['month_sin'].iloc[i] <= 1)
            self.assertTrue(-1 <= data_with_features['month_cos'].iloc[i] <= 1)
            self.assertTrue(-1 <= data_with_features['weekday_sin'].iloc[i] <= 1)
            self.assertTrue(-1 <= data_with_features['weekday_cos'].iloc[i] <= 1)
    
    def test_remove_outliers(self):
        """测试异常值移除功能"""
        # 添加一些极端异常值
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[10, 'close'] = data_with_outliers['close'].mean() * 5
        data_with_outliers.loc[20, 'volume'] = data_with_outliers['volume'].mean() * 10
        
        # 使用Z-score方法移除异常值
        cleaned_zscore = remove_outliers(data_with_outliers, method='zscore', threshold=3.0)
        
        # 验证异常值已被替换
        self.assertNotEqual(cleaned_zscore['close'].iloc[10], data_with_outliers['close'].iloc[10])
        self.assertNotEqual(cleaned_zscore['volume'].iloc[20], data_with_outliers['volume'].iloc[20])
        
        # 使用IQR方法移除异常值
        cleaned_iqr = remove_outliers(data_with_outliers, method='iqr', threshold=1.5)
        
        # 验证异常值已被替换
        self.assertNotEqual(cleaned_iqr['close'].iloc[10], data_with_outliers['close'].iloc[10])
        self.assertNotEqual(cleaned_iqr['volume'].iloc[20], data_with_outliers['volume'].iloc[20])
        
        # 验证没有NaN值（所有异常值都已被替换）
        self.assertEqual(cleaned_zscore.isna().sum().sum(), 0)
        self.assertEqual(cleaned_iqr.isna().sum().sum(), 0)

class TestDataLoading(unittest.TestCase):
    """测试数据加载模块的功能"""
    
    def test_generate_sample_data(self):
        """测试样本数据生成功能"""
        # 生成样本数据
        days = 100
        data = generate_sample_data(days=days, start_price=100, volatility=0.01, trend=0.0001, seed=42)
        
        # 验证生成的行数
        self.assertEqual(len(data), days)
        
        # 验证必要的列存在
        for col in ['date', 'open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, data.columns)
        
        # 验证价格关系
        for i in range(len(data)):
            self.assertLessEqual(data['low'].iloc[i], data['high'].iloc[i])
            self.assertLessEqual(data['low'].iloc[i], data['open'].iloc[i])
            self.assertLessEqual(data['low'].iloc[i], data['close'].iloc[i])
            self.assertGreaterEqual(data['high'].iloc[i], data['open'].iloc[i])
            self.assertGreaterEqual(data['high'].iloc[i], data['close'].iloc[i])
        
        # 验证日期是连续的
        for i in range(1, len(data)):
            self.assertEqual(data['date'].iloc[i] - data['date'].iloc[i-1], timedelta(days=1))
        
        # 验证使用种子生成的数据具有可重复性
        data2 = generate_sample_data(days=days, start_price=100, volatility=0.01, trend=0.0001, seed=42)
        pd.testing.assert_frame_equal(data, data2)
        
        # 验证使用不同种子生成的数据不同
        data3 = generate_sample_data(days=days, start_price=100, volatility=0.01, trend=0.0001, seed=43)
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(data, data3)

if __name__ == '__main__':
    unittest.main()