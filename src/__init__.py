"""
多因子技术分析买入信号评分系统
"""

__version__ = '0.1.0'

# 导入数据加载模块
from .data.data_loader import (
    load_from_yahoo, load_from_csv, load_from_custom_api, generate_sample_data
)

# 导入数据处理模块
from .data.data_processor import (
    clean_price_data, normalize_data, split_train_test, 
    resample_data, add_trading_days, add_date_features, remove_outliers
)

# 导入技术指标计算模块
from .scorer.indicators import (
    calculate_rsi, calculate_bollinger_bands, calculate_moving_averages,
    calculate_volume_indicators, calculate_volatility_indicators, 
    calculate_momentum_indicators, identify_support_levels,
    identify_candlestick_patterns, detect_divergences, has_talib
)

# 导入评分器模块
from .scorer.buy_signal_scorer import BuySignalScorer

# 导入可视化模块
from .visualization.plot_utils import (
    plot_price_with_signals, plot_signal_components, 
    plot_score_history, create_plotly_chart
)

# 版本信息
VERSION = __version__

# 导出所有公共API
__all__ = [
    # 数据加载
    'load_from_yahoo', 'load_from_csv', 'load_from_custom_api', 'generate_sample_data',
    
    # 数据处理
    'clean_price_data', 'normalize_data', 'split_train_test', 'resample_data',
    'add_trading_days', 'add_date_features', 'remove_outliers',
    
    # 技术指标
    'calculate_rsi', 'calculate_bollinger_bands', 'calculate_moving_averages',
    'calculate_volume_indicators', 'calculate_volatility_indicators', 
    'calculate_momentum_indicators', 'identify_support_levels',
    'identify_candlestick_patterns', 'detect_divergences', 'has_talib',
    
    # 评分器
    'BuySignalScorer',
    
    # 可视化
    'plot_price_with_signals', 'plot_signal_components', 
    'plot_score_history', 'create_plotly_chart',
    
    # 版本信息
    'VERSION'
]