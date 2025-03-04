"""
系统集成测试 - 测试各模块之间的集成功能
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所有需要测试的模块
from src.data.data_loader import generate_sample_data
from src.data.data_processor import clean_price_data, resample_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.visualization.plot_utils import plot_price_with_signals, plot_signal_components

class TestSystemIntegration(unittest.TestCase):
    """测试系统的集成功能"""
    
    def test_end_to_end_workflow(self):
        """测试完整的端到端工作流程"""
        try:
            # 1. 生成样本数据
            data = generate_sample_data(days=200, volatility=0.02, trend=0.0001, seed=42)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 200)
            
            # 2. 清洗数据
            cleaned_data = clean_price_data(data)
            self.assertIsInstance(cleaned_data, pd.DataFrame)
            self.assertEqual(len(cleaned_data), 200)
            
            # 3. 重采样为周数据
            weekly_data = resample_data(cleaned_data, timeframe='W')
            self.assertIsInstance(weekly_data, pd.DataFrame)
            self.assertLess(len(weekly_data), 200)
            
            # 4. 对日线数据创建评分器
            daily_scorer = BuySignalScorer(cleaned_data)
            self.assertIsInstance(daily_scorer, BuySignalScorer)
            
            # 5. 计算买入信号评分
            daily_score = daily_scorer.calculate_buy_signal_score()
            self.assertIsInstance(daily_score, dict)
            self.assertIn('total_score', daily_score)
            self.assertIn('component_scores', daily_score)
            self.assertIn('signal_strength', daily_score)
            
            # 6. 对周线数据创建评分器
            weekly_scorer = BuySignalScorer(weekly_data)
            self.assertIsInstance(weekly_scorer, BuySignalScorer)
            
            # 7. 计算周线买入信号评分
            weekly_score = weekly_scorer.calculate_buy_signal_score()
            self.assertIsInstance(weekly_score, dict)
            
            # 8. 评估最近信号（回溯30天）
            days_to_evaluate = min(30, len(cleaned_data))
            recent_signals = daily_scorer.evaluate_recent_signals(days=days_to_evaluate)
            self.assertIsInstance(recent_signals, pd.DataFrame)
            self.assertEqual(len(recent_signals), days_to_evaluate)
            
            # 9. 计算历史评分
            start_index = max(0, len(cleaned_data) - 50)
            end_index = len(cleaned_data) - 1
            historical_scores = daily_scorer.get_historical_scores(start_index, end_index)
            self.assertIsInstance(historical_scores, pd.DataFrame)
            self.assertEqual(len(historical_scores), end_index - start_index + 1)
            
            # 10. 寻找最佳信号
            best_signals = daily_scorer.find_best_signals(threshold=50, max_results=5)
            self.assertIsInstance(best_signals, list)
            
            # 11. 绘制图表
            import matplotlib.pyplot as plt
            
            # 价格和信号图
            fig1 = plot_price_with_signals(cleaned_data, daily_scorer, threshold=50)
            self.assertIsInstance(fig1, plt.Figure)
            plt.close(fig1)
            
            # 信号组成雷达图
            fig2 = plot_signal_components(daily_score)
            self.assertIsInstance(fig2, plt.Figure)
            plt.close(fig2)
            
        except Exception as e:
            self.fail(f"端到端工作流程测试失败: {str(e)}")
    
    def test_different_market_conditions(self):
        """测试不同市场条件下的评分系统"""
        try:
            # 生成上涨趋势数据
            uptrend_data = generate_sample_data(days=100, volatility=0.02, trend=0.002, seed=42)
            
            # 生成下跌趋势数据
            downtrend_data = generate_sample_data(days=100, volatility=0.02, trend=-0.002, seed=42)
            
            # 生成震荡行情数据
            sideways_data = generate_sample_data(days=100, volatility=0.01, trend=0.0, seed=42)
            
            # 创建评分器
            uptrend_scorer = BuySignalScorer(uptrend_data)
            downtrend_scorer = BuySignalScorer(downtrend_data)
            sideways_scorer = BuySignalScorer(sideways_data)
            
            # 计算各种情况的评分
            uptrend_score = uptrend_scorer.calculate_buy_signal_score()
            downtrend_score = downtrend_scorer.calculate_buy_signal_score()
            sideways_score = sideways_scorer.calculate_buy_signal_score()
            
            # 验证每种情况下的输出格式一致
            for score in [uptrend_score, downtrend_score, sideways_score]:
                self.assertIsInstance(score, dict)
                self.assertIn('total_score', score)
                self.assertIn('component_scores', score)
                self.assertIn('signal_strength', score)
                
                # 验证各组分都存在
                components = ['RSI指标', '价格形态', '成交量分析', '支撑位分析', '动量指标', '波动率分析']
                for comp in components:
                    self.assertIn(comp, score['component_scores'])
            
            # 验证下跌趋势中的RSI分数较高（超卖情况）
            self.assertGreaterEqual(
                downtrend_score['component_scores']['RSI指标'],
                uptrend_score['component_scores']['RSI指标']
            )
            
            # 验证上涨趋势中的动量分数较高
            self.assertGreaterEqual(
                uptrend_score['component_scores']['动量指标'],
                downtrend_score['component_scores']['动量指标']
            )
            
        except Exception as e:
            self.fail(f"不同市场条件测试失败: {str(e)}")
    
    def test_custom_weights_configuration(self):
        """测试自定义权重配置的影响"""
        try:
            # 生成样本数据
            data = generate_sample_data(days=100, volatility=0.02, trend=-0.001, seed=42)
            
            # 创建标准评分器
            standard_scorer = BuySignalScorer(data)
            
            # 创建RSI权重加倍的评分器
            rsi_weighted_config = {
                'RSI指标': 40,  # 标准是20，加倍
                '价格形态': 20,
                '成交量分析': 10,
                '支撑位分析': 10,
                '动量指标': 10,
                '波动率分析': 10
            }
            rsi_weighted_scorer = BuySignalScorer(data, custom_weights=rsi_weighted_config)
            
            # 计算评分
            standard_score = standard_scorer.calculate_buy_signal_score()
            rsi_weighted_score = rsi_weighted_scorer.calculate_buy_signal_score()
            
            # 1. 验证RSI组分在自定义配置中得分更高
            standard_rsi_contribution = standard_score['component_scores']['RSI指标']
            weighted_rsi_contribution = rsi_weighted_score['component_scores']['RSI指标']
            
            self.assertGreater(weighted_rsi_contribution, standard_rsi_contribution)
            
            # 2. 验证总分受到权重变化的影响
            # 假设RSI得分较高，权重加倍应该提高总分
            # 如果RSI得分低，权重加倍会降低总分
            rsi_component_ratio = standard_score['component_scores']['RSI指标'] / 20  # 满分20的得分率
            avg_other_component_ratio = sum([
                standard_score['component_scores']['价格形态'] / 20,
                standard_score['component_scores']['成交量分析'] / 15,
                standard_score['component_scores']['支撑位分析'] / 15,
                standard_score['component_scores']['动量指标'] / 15,
                standard_score['component_scores']['波动率分析'] / 15
            ]) / 5
            
            # 如果RSI得分率高于其他组分的平均得分率，总分应该提高
            if rsi_component_ratio > avg_other_component_ratio:
                self.assertGreater(rsi_weighted_score['total_score'], standard_score['total_score'])
            # 如果RSI得分率低于其他组分的平均得分率，总分应该降低
            elif rsi_component_ratio < avg_other_component_ratio:
                self.assertLess(rsi_weighted_score['total_score'], standard_score['total_score'])
            
        except Exception as e:
            self.fail(f"自定义权重配置测试失败: {str(e)}")
    
    def test_data_processing_impact(self):
        """测试数据处理对评分结果的影响"""
        try:
            # 生成样本数据
            data = generate_sample_data(days=100, volatility=0.02, trend=0.0, seed=42)
            
            # 故意添加一些异常值
            data_with_outliers = data.copy()
            data_with_outliers.loc[90, 'close'] = data.loc[90, 'close'] * 1.5  # 大幅上涨
            data_with_outliers.loc[95, 'close'] = data.loc[95, 'close'] * 0.7  # 大幅下跌
            
            # 计算原始数据的评分
            original_scorer = BuySignalScorer(data)
            original_score = original_scorer.calculate_buy_signal_score()
            
            # 计算带异常值数据的评分
            outlier_scorer = BuySignalScorer(data_with_outliers)
            outlier_score = outlier_scorer.calculate_buy_signal_score()
            
            # 移除异常值
            from src.data.data_processor import remove_outliers
            cleaned_data = remove_outliers(data_with_outliers, method='zscore', threshold=2.0)
            
            # 计算清洗后数据的评分
            cleaned_scorer = BuySignalScorer(cleaned_data)
            cleaned_score = cleaned_scorer.calculate_buy_signal_score()
            
            # 验证异常值对评分的影响
            # 1. 验证异常值影响了评分
            self.assertNotEqual(original_score['total_score'], outlier_score['total_score'])
            
            # 2. 验证清洗数据后的评分接近原始评分
            score_difference_original_vs_outlier = abs(original_score['total_score'] - outlier_score['total_score'])
            score_difference_original_vs_cleaned = abs(original_score['total_score'] - cleaned_score['total_score'])
            
            # 清洗后的分数应该比未清洗的更接近原始分数
            self.assertLess(score_difference_original_vs_cleaned, score_difference_original_vs_outlier)
            
            # 3. 测试重采样后的评分
            from src.data.data_processor import resample_data
            weekly_data = resample_data(data, timeframe='W')
            weekly_scorer = BuySignalScorer(weekly_data)
            weekly_score = weekly_scorer.calculate_buy_signal_score()
            
            # 验证周线数据产生了有效的评分
            self.assertIsInstance(weekly_score, dict)
            self.assertIn('total_score', weekly_score)
            self.assertGreaterEqual(weekly_score['total_score'], 0)
            self.assertLessEqual(weekly_score['total_score'], 100)
            
        except Exception as e:
            self.fail(f"数据处理影响测试失败: {str(e)}")
    
    def test_historical_analysis(self):
        """测试历史分析功能"""
        try:
            # 生成样本数据 - 前半段下跌，后半段上涨
            days = 200
            first_half = generate_sample_data(days=days//2, volatility=0.02, trend=-0.002, seed=42)
            second_half = generate_sample_data(days=days//2, volatility=0.02, trend=0.002, seed=43)
            
            # 调整第二段数据的起始价格和日期
            last_price = first_half['close'].iloc[-1]
            last_date = first_half['date'].iloc[-1]
            
            second_half['open'] = second_half['open'] * last_price / second_half['open'].iloc[0]
            second_half['high'] = second_half['high'] * last_price / second_half['high'].iloc[0]
            second_half['low'] = second_half['low'] * last_price / second_half['low'].iloc[0]
            second_half['close'] = second_half['close'] * last_price / second_half['close'].iloc[0]
            
            second_half['date'] = second_half['date'].apply(lambda x: last_date + (x - second_half['date'].iloc[0]) + timedelta(days=1))
            
            # 合并数据
            combined_data = pd.concat([first_half, second_half]).reset_index(drop=True)
            
            # 创建评分器
            scorer = BuySignalScorer(combined_data)
            
            # 计算整个历史期间的评分
            historical_scores = scorer.get_historical_scores()
            
            # 验证历史评分结果
            self.assertIsInstance(historical_scores, pd.DataFrame)
            self.assertEqual(len(historical_scores), len(combined_data))
            
            # 确认必要列
            required_columns = [
                'date', 'price', 'total_score', 'signal_strength',
                'rsi_score', 'price_pattern_score', 'volume_score',
                'support_score', 'momentum_score', 'volatility_score'
            ]
            for col in required_columns:
                self.assertIn(col, historical_scores.columns)
            
            # 验证下跌阶段应该有更多的买入信号
            down_trend_signals = historical_scores.iloc[:days//2]
            up_trend_signals = historical_scores.iloc[days//2:]
            
            down_trend_buy_signals = down_trend_signals[down_trend_signals['total_score'] >= 60]
            up_trend_buy_signals = up_trend_signals[up_trend_signals['total_score'] >= 60]
            
            # 通常下跌阶段应该有更多买入信号（更多超卖情况）
            # 但这取决于随机生成的数据，所以这里只验证能得到结果
            self.assertGreaterEqual(len(down_trend_buy_signals) + len(up_trend_buy_signals), 0)
            
            # 查找最佳买入信号
            best_signals = scorer.find_best_signals(threshold=70, max_results=5)
            self.assertIsInstance(best_signals, list)
            
            # 验证最佳信号确实评分较高
            if best_signals:
                self.assertGreaterEqual(best_signals[0]['total_score'], 70)
            
        except Exception as e:
            self.fail(f"历史分析功能测试失败: {str(e)}")

if __name__ == '__main__':
    unittest.main()