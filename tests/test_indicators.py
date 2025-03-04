"""
技术指标计算模块的单元测试
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
from src.scorer.indicators import (
    calculate_rsi, calculate_bollinger_bands, calculate_moving_averages,
    calculate_volume_indicators, calculate_volatility_indicators, 
    calculate_momentum_indicators, identify_support_levels,
    identify_candlestick_patterns, detect_divergences
)

class TestIndicatorsCalculation(unittest.TestCase):
    """测试各类技术指标的计算"""
    
    def setUp(self):
        """测试前准备工作"""
        # 生成测试用的样本数据
        self.sample_data = generate_sample_data(days=200, volatility=0.02, trend=0.0001, seed=42)
    
    def test_rsi_calculation(self):
        """测试RSI计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_rsi(self.sample_data, periods=[6, 14, 28], use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('rsi_6', result.columns)
        self.assertIn('rsi_14', result.columns)
        self.assertIn('rsi_28', result.columns)
        
        # 验证RSI值范围是否在0-100之间
        for period in [6, 14, 28]:
            column = f'rsi_{period}'
            # 跳过前period个可能的NaN值
            values = result[column].dropna().values
            self.assertTrue(all(0 <= val <= 100 for val in values))
        
        # 验证RSI计算是否合理
        # RSI应该对价格变化做出反应
        price_change = self.sample_data['close'].pct_change()
        # 当价格连续上涨时，RSI值应该较高
        high_rsi_periods = price_change.rolling(5).sum() > 0
        for i in range(30, len(result)):
            if high_rsi_periods.iloc[i]:
                self.assertGreater(result['rsi_14'].iloc[i], 50, 
                                  f"连续上涨期间RSI应大于50，但在索引{i}处是{result['rsi_14'].iloc[i]}")
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_bollinger_bands(self.sample_data, period=20, std_devs=[2, 3], use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('bb_upper', result.columns)
        self.assertIn('bb_middle', result.columns)
        self.assertIn('bb_lower', result.columns)
        self.assertIn('bb_upper_3std', result.columns)
        self.assertIn('bb_middle_3std', result.columns)
        self.assertIn('bb_lower_3std', result.columns)
        
        # 验证布林带基本关系
        # 上轨 > 中轨 > 下轨
        for i in range(20, len(result)):  # 跳过前20个可能的NaN值
            self.assertGreater(result['bb_upper'].iloc[i], result['bb_middle'].iloc[i])
            self.assertGreater(result['bb_middle'].iloc[i], result['bb_lower'].iloc[i])
            self.assertGreater(result['bb_upper_3std'].iloc[i], result['bb_upper'].iloc[i])
            self.assertLess(result['bb_lower_3std'].iloc[i], result['bb_lower'].iloc[i])
        
        # 验证中轨是否等于20日均线
        for i in range(20, len(result)):
            expected_sma = self.sample_data['close'].iloc[i-20:i].mean()
            self.assertAlmostEqual(result['bb_middle'].iloc[i], expected_sma, places=6)
    
    def test_moving_averages_calculation(self):
        """测试移动平均线计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_moving_averages(
            self.sample_data, periods=[10, 20, 50], 
            types=['sma', 'ema'], use_talib=False
        )
        
        # 验证结果列是否存在
        self.assertIn('sma_10', result.columns)
        self.assertIn('sma_20', result.columns)
        self.assertIn('sma_50', result.columns)
        self.assertIn('ema_10', result.columns)
        self.assertIn('ema_20', result.columns)
        self.assertIn('ema_50', result.columns)
        
        # 验证SMA计算是否正确
        for i in range(50, len(result), 10):  # 每隔10个点检查一次
            for period in [10, 20, 50]:
                expected_sma = self.sample_data['close'].iloc[i-period:i].mean()
                self.assertAlmostEqual(result[f'sma_{period}'].iloc[i], expected_sma, places=6)
        
        # 验证EMA与SMA的关系
        # 在上涨趋势中，EMA应高于SMA；在下跌趋势中，EMA应低于SMA
        for i in range(60, len(result), 10):
            trend = result['close'].iloc[i-10:i].mean() - result['close'].iloc[i-20:i-10].mean()
            if trend > 0:  # 上涨趋势
                self.assertGreater(result['ema_10'].iloc[i], result['sma_10'].iloc[i])
            elif trend < 0:  # 下跌趋势
                self.assertLess(result['ema_10'].iloc[i], result['sma_10'].iloc[i])
    
    def test_volume_indicators_calculation(self):
        """测试成交量指标计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_volume_indicators(self.sample_data, periods=[10, 20], use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('volume_sma_10', result.columns)
        self.assertIn('volume_sma_20', result.columns)
        self.assertIn('relative_volume', result.columns)
        self.assertIn('volume_volatility', result.columns)
        self.assertIn('obv', result.columns)
        
        # 验证成交量平均线计算是否正确
        for i in range(20, len(result), 10):
            expected_volume_sma_10 = self.sample_data['volume'].iloc[i-10:i].mean()
            self.assertAlmostEqual(result['volume_sma_10'].iloc[i], expected_volume_sma_10, places=6)
            
            expected_volume_sma_20 = self.sample_data['volume'].iloc[i-20:i].mean()
            self.assertAlmostEqual(result['volume_sma_20'].iloc[i], expected_volume_sma_20, places=6)
        
        # 验证相对成交量计算是否正确
        for i in range(20, len(result), 10):
            expected_relative_volume = self.sample_data['volume'].iloc[i] / result['volume_sma_20'].iloc[i]
            self.assertAlmostEqual(result['relative_volume'].iloc[i], expected_relative_volume, places=6)
    
    def test_volatility_indicators_calculation(self):
        """测试波动率指标计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_volatility_indicators(self.sample_data, use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('atr', result.columns)
        self.assertIn('relative_atr', result.columns)
        self.assertIn('atr_change', result.columns)
        
        # 验证ATR值是否大于0
        for i in range(14, len(result)):  # 跳过前14个可能的NaN值
            self.assertGreater(result['atr'].iloc[i], 0)
        
        # 验证相对ATR计算是否正确
        for i in range(14, len(result), 10):
            expected_relative_atr = result['atr'].iloc[i] / result['close'].iloc[i] * 100
            self.assertAlmostEqual(result['relative_atr'].iloc[i], expected_relative_atr, places=6)
    
    def test_momentum_indicators_calculation(self):
        """测试动量指标计算函数"""
        # 测试内部实现（不使用TA-Lib）
        result = calculate_momentum_indicators(self.sample_data, use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('slowk', result.columns)
        self.assertIn('slowd', result.columns)
        self.assertIn('willr', result.columns)
        self.assertIn('cci', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_hist', result.columns)
        
        # 验证慢速随机指标K和D值是否在0-100之间
        for i in range(20, len(result)):  # 跳过前几个可能的NaN值
            self.assertTrue(0 <= result['slowk'].iloc[i] <= 100)
            self.assertTrue(0 <= result['slowd'].iloc[i] <= 100)
        
        # 验证威廉指标取值是否在-100~0之间
        for i in range(20, len(result)):
            self.assertTrue(-100 <= result['willr'].iloc[i] <= 0)
        
        # 验证MACD柱状图的计算
        for i in range(30, len(result)):
            expected_hist = result['macd'].iloc[i] - result['macd_signal'].iloc[i]
            self.assertAlmostEqual(result['macd_hist'].iloc[i], expected_hist, places=6)
    
    def test_support_levels_identification(self):
        """测试支撑位识别函数"""
        support_levels, fib_levels = identify_support_levels(self.sample_data, lookback=100, n_points=2)
        
        # 验证支撑位和斐波那契回调位是否生成
        self.assertIsInstance(support_levels, list)
        self.assertIsInstance(fib_levels, dict)
        
        # 验证斐波那契回调位是否包含所有需要的级别
        expected_fib_keys = ['fib_0', 'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786', 'fib_1']
        for key in expected_fib_keys:
            self.assertIn(key, fib_levels)
        
        # 验证斐波那契级别关系
        if fib_levels:
            self.assertLess(fib_levels['fib_0'], fib_levels['fib_0.236'])
            self.assertLess(fib_levels['fib_0.236'], fib_levels['fib_0.382'])
            self.assertLess(fib_levels['fib_0.382'], fib_levels['fib_0.5'])
            self.assertLess(fib_levels['fib_0.5'], fib_levels['fib_0.618'])
            self.assertLess(fib_levels['fib_0.618'], fib_levels['fib_0.786'])
            self.assertLess(fib_levels['fib_0.786'], fib_levels['fib_1'])
    
    def test_candlestick_patterns_identification(self):
        """测试蜡烛图形态识别函数"""
        result = identify_candlestick_patterns(self.sample_data, use_talib=False)
        
        # 验证结果列是否存在
        self.assertIn('body_size', result.columns)
        self.assertIn('total_range', result.columns)
        self.assertIn('upper_shadow', result.columns)
        self.assertIn('lower_shadow', result.columns)
        self.assertIn('long_lower_shadow', result.columns)
        self.assertIn('hammer', result.columns)
        self.assertIn('doji', result.columns)
        
        # 验证蜡烛图相关属性的计算
        for i in range(len(result)):
            # body_size = |close - open|
            expected_body_size = abs(result['close'].iloc[i] - result['open'].iloc[i])
            self.assertAlmostEqual(result['body_size'].iloc[i], expected_body_size, places=6)
            
            # total_range = high - low
            expected_total_range = result['high'].iloc[i] - result['low'].iloc[i]
            self.assertAlmostEqual(result['total_range'].iloc[i], expected_total_range, places=6)
            
            # upper_shadow = high - max(open, close)
            expected_upper_shadow = result['high'].iloc[i] - max(result['open'].iloc[i], result['close'].iloc[i])
            self.assertAlmostEqual(result['upper_shadow'].iloc[i], expected_upper_shadow, places=6)
            
            # lower_shadow = min(open, close) - low
            expected_lower_shadow = min(result['open'].iloc[i], result['close'].iloc[i]) - result['low'].iloc[i]
            self.assertAlmostEqual(result['lower_shadow'].iloc[i], expected_lower_shadow, places=6)
    
    def test_divergences_detection(self):
        """测试背离检测函数"""
        # 先计算RSI和MACD
        data = calculate_rsi(self.sample_data, use_talib=False)
        data = calculate_momentum_indicators(data, use_talib=False)
        data = calculate_volume_indicators(data, use_talib=False)
        
        # 检测背离
        result = detect_divergences(data, lookback=20)
        
        # 验证结果列是否存在
        self.assertIn('rsi_bullish_divergence', result.columns)
        self.assertIn('macd_bullish_divergence', result.columns)
        self.assertIn('obv_bullish_divergence', result.columns)
        
        # 验证背离标记是否为布尔值
        self.assertTrue(all(isinstance(val, (bool, np.bool_)) for val in result['rsi_bullish_divergence']))
        self.assertTrue(all(isinstance(val, (bool, np.bool_)) for val in result['macd_bullish_divergence']))
        self.assertTrue(all(isinstance(val, (bool, np.bool_)) for val in result['obv_bullish_divergence']))

if __name__ == '__main__':
    unittest.main()