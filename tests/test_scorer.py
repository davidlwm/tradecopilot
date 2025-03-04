"""
买入信号评分系统的单元测试
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
from src.scorer.buy_signal_scorer import BuySignalScorer

class TestBuySignalScorer(unittest.TestCase):
    """测试买入信号评分系统"""
    
    def setUp(self):
        """测试前准备工作"""
        # 生成测试用的样本数据
        self.sample_data = generate_sample_data(days=200, volatility=0.02, trend=0.0001, seed=42)
        
        # 创建评分器
        self.scorer = BuySignalScorer(self.sample_data)
    
    def test_initialization(self):
        """测试评分器初始化"""
        # 验证评分器是否正确创建
        self.assertIsInstance(self.scorer, BuySignalScorer)
        
        # 验证指标是否已计算
        self.assertIn('rsi_14', self.scorer.data.columns)
        self.assertIn('sma_50', self.scorer.data.columns)
        self.assertIn('macd', self.scorer.data.columns)
        self.assertIn('obv', self.scorer.data.columns)
    
    def test_custom_weights(self):
        """测试自定义权重功能"""
        # 创建带有自定义权重的评分器
        custom_weights = {
            'RSI指标': 30,  # 调高RSI的权重
            '价格形态': 20,
            '成交量分析': 15,
            '支撑位分析': 15,
            '动量指标': 10,  # 调低动量指标的权重
            '波动率分析': 10   # 调低波动率指标的权重
        }
        
        custom_scorer = BuySignalScorer(self.sample_data, custom_weights=custom_weights)
        
        # 验证权重是否正确应用
        self.assertEqual(custom_scorer.weights['RSI指标'], 30)
        self.assertEqual(custom_scorer.weights['动量指标'], 10)
        self.assertEqual(custom_scorer.weights['波动率分析'], 10)
        
        # 验证总权重是否为100
        self.assertEqual(sum(custom_scorer.weights.values()), 100)
    
    def test_component_scoring(self):
        """测试各组件评分函数"""
        # 获取最后一个数据点的索引
        idx = len(self.sample_data) - 1
        
        # 测试RSI评分
        rsi_score = self.scorer._score_rsi(idx)
        self.assertGreaterEqual(rsi_score, 0)
        self.assertLessEqual(rsi_score, 20)
        
        # 测试价格形态评分
        price_score = self.scorer._score_price_pattern(idx)
        self.assertGreaterEqual(price_score, 0)
        self.assertLessEqual(price_score, 20)
        
        # 测试成交量评分
        volume_score = self.scorer._score_volume(idx)
        self.assertGreaterEqual(volume_score, 0)
        self.assertLessEqual(volume_score, 15)
        
        # 测试支撑位评分
        support_score = self.scorer._score_support_resistance(idx)
        self.assertGreaterEqual(support_score, 0)
        self.assertLessEqual(support_score, 15)
        
        # 测试动量评分
        momentum_score = self.scorer._score_momentum(idx)
        self.assertGreaterEqual(momentum_score, 0)
        self.assertLessEqual(momentum_score, 15)
        
        # 测试波动率评分
        volatility_score = self.scorer._score_volatility(idx)
        self.assertGreaterEqual(volatility_score, 0)
        self.assertLessEqual(volatility_score, 15)
    
    def test_total_score_calculation(self):
        """测试总分计算"""
        # 计算最后一个数据点的总分
        score_data = self.scorer.calculate_buy_signal_score()
        
        # 验证分数在0-100范围内
        self.assertGreaterEqual(score_data['total_score'], 0)
        self.assertLessEqual(score_data['total_score'], 100)
        
        # 验证日期和价格字段存在
        self.assertIn('date', score_data)
        self.assertIn('price', score_data)
        
        # 验证所有组件分数都存在
        self.assertIn('component_scores', score_data)
        self.assertEqual(len(score_data['component_scores']), 6)
        
        # 验证信号强度字段存在
        self.assertIn('signal_strength', score_data)
        
        # 验证总分等于各组件分数之和
        component_sum = sum(score_data['component_scores'].values())
        self.assertAlmostEqual(score_data['total_score'], component_sum, places=6)
    
    def test_signal_strength_mapping(self):
        """测试信号强度映射"""
        # 测试各个分数区间的信号强度
        self.assertEqual(self.scorer._get_signal_strength(90), "极强买入信号")
        self.assertEqual(self.scorer._get_signal_strength(75), "强买入信号")
        self.assertEqual(self.scorer._get_signal_strength(65), "中强买入信号")
        self.assertEqual(self.scorer._get_signal_strength(55), "中等买入信号")
        self.assertEqual(self.scorer._get_signal_strength(45), "弱买入信号")
        self.assertEqual(self.scorer._get_signal_strength(35), "极弱买入信号")
        self.assertEqual(self.scorer._get_signal_strength(25), "无买入信号")
    
    def test_recent_signals_evaluation(self):
        """测试最近信号评估函数"""
        # 评估最近10天的信号
        recent_signals = self.scorer.evaluate_recent_signals(days=10)
        
        # 验证返回的DataFrame长度
        self.assertEqual(len(recent_signals), 10)
        
        # 验证DataFrame包含必要的列
        self.assertIn('date', recent_signals.columns)
        self.assertIn('price', recent_signals.columns)
        self.assertIn('total_score', recent_signals.columns)
        self.assertIn('signal_strength', recent_signals.columns)
        
        # 验证日期是按照时间顺序排列的
        dates = pd.to_datetime(recent_signals['date'])
        self.assertTrue(all(dates.iloc[i] <= dates.iloc[i+1] for i in range(len(dates)-1)))
    
    def test_historical_scores_calculation(self):
        """测试历史评分计算函数"""
        # 计算多个日期点的评分
        start_index = 50
        end_index = 60
        historical_scores = self.scorer.get_historical_scores(start_index, end_index)
        
        # 验证返回的DataFrame长度
        self.assertEqual(len(historical_scores), end_index - start_index + 1)
        
        # 验证包含所有必要的列
        self.assertIn('date', historical_scores.columns)
        self.assertIn('price', historical_scores.columns)
        self.assertIn('total_score', historical_scores.columns)
        self.assertIn('signal_strength', historical_scores.columns)
        
        # 验证包含各组件的分数列
        self.assertIn('rsi_score', historical_scores.columns)
        self.assertIn('price_pattern_score', historical_scores.columns)
        self.assertIn('volume_score', historical_scores.columns)
        self.assertIn('support_score', historical_scores.columns)
        self.assertIn('momentum_score', historical_scores.columns)
        self.assertIn('volatility_score', historical_scores.columns)
    
    def test_find_best_signals(self):
        """测试最佳信号查找函数"""
        # 查找评分高于阈值的信号点
        threshold = 60
        max_results = 3
        best_signals = self.scorer.find_best_signals(threshold=threshold, max_results=max_results)
        
        # 验证结果是否为列表
        self.assertIsInstance(best_signals, list)
        
        # 验证结果数量不超过max_results
        self.assertLessEqual(len(best_signals), max_results)
        
        # 验证所有返回的信号分数都大于等于阈值
        for signal in best_signals:
            self.assertGreaterEqual(signal['total_score'], threshold)
        
        # 验证返回的信号按分数降序排列
        if len(best_signals) >= 2:
            for i in range(len(best_signals) - 1):
                self.assertGreaterEqual(
                    best_signals[i]['total_score'],
                    best_signals[i+1]['total_score']
                )
    
    def test_specific_market_conditions(self):
        """测试特定市场条件下的评分"""
        # 模拟RSI超卖的情况
        data = self.sample_data.copy()
        data['rsi_6'] = 20  # 设置RSI为超卖状态
        data['rsi_14'] = 25
        data['rsi_28'] = 30
        
        # 创建新的评分器
        custom_scorer = BuySignalScorer(data)
        
        # 计算分数
        score_data = custom_scorer.calculate_buy_signal_score()
        
        # 验证RSI分数较高
        self.assertGreater(score_data['component_scores']['RSI指标'], 10)
        
        # 模拟布林带超卖情况
        data = self.sample_data.copy()
        # 设置收盘价低于布林带下轨
        current_close = data['close'].iloc[-1]
        data['bb_lower'] = current_close * 1.05
        data['bb_middle'] = current_close * 1.15
        data['bb_upper'] = current_close * 1.25
        
        # 创建新的评分器
        custom_scorer = BuySignalScorer(data)
        
        # 计算分数
        score_data = custom_scorer.calculate_buy_signal_score()
        
        # 验证价格形态分数较高
        self.assertGreater(score_data['component_scores']['价格形态'], 5)
        
        # 模拟MACD底背离情况
        data = self.sample_data.copy()
        data['macd_bullish_divergence'] = True
        
        # 创建新的评分器
        custom_scorer = BuySignalScorer(data)
        
        # 计算分数
        score_data = custom_scorer.calculate_buy_signal_score()
        
        # 验证动量指标分数较高
        self.assertGreater(score_data['component_scores']['动量指标'], 5)

if __name__ == '__main__':
    unittest.main()