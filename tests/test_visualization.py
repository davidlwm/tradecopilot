"""
可视化模块的单元测试
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import generate_sample_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.visualization.plot_utils import (
    plot_price_with_signals, plot_signal_components, 
    plot_score_history, create_plotly_chart
)

class TestVisualization(unittest.TestCase):
    """测试可视化模块的功能"""
    
    def setUp(self):
        """测试前准备工作"""
        # 生成测试用的样本数据
        self.sample_data = generate_sample_data(days=100, volatility=0.02, trend=0.0001, seed=42)
        
        # 创建评分器
        self.scorer = BuySignalScorer(self.sample_data)
        
        # 计算评分
        self.score_data = self.scorer.calculate_buy_signal_score()
        
        # 评估最近10天的信号
        self.recent_signals = self.scorer.evaluate_recent_signals(days=10)
    
    def test_plot_price_with_signals(self):
        """测试价格和信号图表绘制功能"""
        try:
            # 绘制图表
            fig = plot_price_with_signals(self.sample_data, self.scorer, threshold=50)
            
            # 验证是否成功创建了图表
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图表有正确的子图数量
            self.assertEqual(len(fig.axes), 2)
            
            # 关闭图表，释放资源
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"绘制价格与信号图表时出错: {str(e)}")
    
    def test_plot_signal_components(self):
        """测试信号组成雷达图绘制功能"""
        try:
            # 绘制图表
            fig = plot_signal_components(self.score_data)
            
            # 验证是否成功创建了图表
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图表有正确的子图数量（极坐标图）
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(fig.axes[0].name, 'polar')
            
            # 关闭图表，释放资源
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"绘制信号组成雷达图时出错: {str(e)}")
    
    def test_plot_score_history(self):
        """测试评分历史趋势图绘制功能"""
        try:
            # 绘制图表
            fig = plot_score_history(self.recent_signals)
            
            # 验证是否成功创建了图表
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图表有正确的子图数量
            self.assertEqual(len(fig.axes), 2)
            
            # 关闭图表，释放资源
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"绘制评分历史趋势图时出错: {str(e)}")
    
    def test_create_plotly_chart(self):
        """测试创建Plotly交互式图表功能"""
        try:
            # 导入plotly模块（如果安装了）
            import importlib
            plotly_spec = importlib.util.find_spec("plotly")
            
            if plotly_spec is not None:
                # 创建图表
                fig = create_plotly_chart(self.sample_data, self.scorer, threshold=50)
                
                # 验证是否成功创建了图表
                self.assertIsInstance(fig, object)
                self.assertTrue(hasattr(fig, 'update_layout'))
                
            else:
                # 如果未安装plotly，跳过测试
                self.skipTest("未安装plotly库，跳过此测试")
                
        except ImportError:
            # 如果导入失败，跳过测试
            self.skipTest("未安装plotly库，跳过此测试")
            
        except Exception as e:
            self.fail(f"创建Plotly图表时出错: {str(e)}")
    
    def test_visualization_edge_cases(self):
        """测试可视化模块的边缘情况"""
        try:
            # 测试空数据
            empty_data = self.sample_data.iloc[0:0]
            
            with self.assertRaises(Exception):
                # 绘制空数据的图表应该抛出异常
                plot_price_with_signals(empty_data, None)
            
            # 测试没有评分器的情况
            fig = plot_price_with_signals(self.sample_data, None)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
            
            # 测试非常短的数据
            short_data = self.sample_data.iloc[:5]
            fig = plot_price_with_signals(short_data, None)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"测试可视化边缘情况时出错: {str(e)}")

if __name__ == '__main__':
    unittest.main()