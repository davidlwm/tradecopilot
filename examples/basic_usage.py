"""
多因子买入信号评分系统 - 基本使用示例
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需模块
from src.data.data_loader import load_from_yahoo, generate_sample_data
from src.data.data_processor import clean_price_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.visualization.plot_utils import plot_price_with_signals, plot_signal_components, plot_score_history

def main():
    """运行基本使用示例"""
    print("多因子买入信号评分系统 - 基本使用示例")
    print("-" * 50)
    
    # 从Yahoo Finance加载数据
    print("\n1. 从Yahoo Finance加载苹果公司(AAPL)股票数据...")
    try:
        data = load_from_yahoo("AAPL", period="1y", interval="1d")
        print(f"加载了 {len(data)} 条数据")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        print("生成样本数据用于演示...")
        data = generate_sample_data(days=252)
        print(f"生成了 {len(data)} 条样本数据")
    
    # 数据清洗
    print("\n2. 数据清洗...")
    data = clean_price_data(data)
    print("数据清洗完成")
    
    # 创建买入信号评分器
    print("\n3. 创建买入信号评分器...")
    scorer = BuySignalScorer(data)
    
    # 计算最新的买入信号得分
    print("\n4. 计算最新的买入信号得分...")
    latest_score = scorer.calculate_buy_signal_score()
    
    print(f"\n日期: {latest_score['date'].strftime('%Y-%m-%d')}")
    print(f"价格: {latest_score['price']:.2f}")
    print(f"总得分: {latest_score['total_score']:.2f}/100")
    print(f"信号强度: {latest_score['signal_strength']}")
    
    print("\n各组成部分得分:")
    for component, score in latest_score['component_scores'].items():
        max_score = 20 if component in ['RSI指标', '价格形态'] else 15
        print(f"  - {component}: {score:.2f}/{max_score} ({score/max_score*100:.1f}%)")
    
    # 评估最近30天的信号
    print("\n5. 评估最近30天的信号...")
    days_to_evaluate = min(30, len(data))
    recent_signals = scorer.evaluate_recent_signals(days=days_to_evaluate)
    print(f"最近{days_to_evaluate}天中，得分 >= 50的天数: {len(recent_signals[recent_signals['total_score'] >= 50])}")
    print(f"最近{days_to_evaluate}天中，得分 >= 70的天数: {len(recent_signals[recent_signals['total_score'] >= 70])}")
    
    # 绘制价格图表和买入信号
    print("\n6. 生成可视化图表...")
    
    try:
        # 价格与信号图
        price_fig = plot_price_with_signals(data, scorer, threshold=50)
        price_fig.savefig("price_with_signals.png")
        print("已保存价格与信号图表: price_with_signals.png")
        
        # 信号组成雷达图
        radar_fig = plot_signal_components(latest_score)
        radar_fig.savefig("signal_components.png")
        print("已保存信号组成雷达图: signal_components.png")
        
        # 信号历史趋势图
        history_fig = plot_score_history(recent_signals)
        history_fig.savefig("score_history.png")
        print("已保存信号历史趋势图: score_history.png")
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
    
    print("\n示例运行完成!")

if __name__ == "__main__":
    main()