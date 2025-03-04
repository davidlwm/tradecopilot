"""
多因子买入信号评分系统 - 测试套件
"""

# 导入所有测试模块，方便批量运行
from .test_data import TestDataProcessing, TestDataLoading
from .test_indicators import TestIndicatorsCalculation
from .test_scorer import TestBuySignalScorer
from .test_visualization import TestVisualization
from .test_integration import TestSystemIntegration