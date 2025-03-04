"""
可视化工具模块 - 提供绘制图表的功能
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_price_with_signals(data, scorer, figsize=(15, 10), threshold=50):
    """
    绘制价格图表及买入信号
    
    参数:
    data (pandas.DataFrame): 价格数据
    scorer (BuySignalScorer): 信号评分器对象
    figsize (tuple): 图表大小
    threshold (float): 信号阈值，只显示得分高于此值的信号
    
    返回:
    matplotlib.figure.Figure: 生成的图表
    """
    # 计算最近所有数据点的信号强度
    signals = scorer.evaluate_recent_signals(days=len(data))
    
    # 筛选出高于阈值的信号
    strong_signals = signals[signals['total_score'] >= threshold]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制价格
    ax.plot(data['date'], data['close'], label='收盘价', color='blue')
    
    # 绘制移动平均线
    if 'sma_20' in data.columns:
        ax.plot(data['date'], data['sma_20'], label='20日均线', color='orange', alpha=0.7)
    if 'sma_50' in data.columns:
        ax.plot(data['date'], data['sma_50'], label='50日均线', color='green', alpha=0.7)
    
    # 在图表上标记买入信号
    for _, signal in strong_signals.iterrows():
        ax.scatter(signal['date'], signal['price'], 
                   color='red', marker='^', s=100, zorder=5,
                   label=f"买入信号 ({signal['total_score']:.1f})" if _ == 0 else "")
        
        # 添加注释
        ax.annotate(f"{signal['total_score']:.1f}", 
                    (signal['date'], signal['price']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color='red')
    
    # 美化图表
    ax.set_title('价格走势与买入信号', fontsize=15)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_signal_components(signal_data, figsize=(15, 8)):
    """
    绘制信号组成部分的雷达图
    
    参数:
    signal_data (dict): 包含组件分数的信号数据
    figsize (tuple): 图表大小
    
    返回:
    matplotlib.figure.Figure: 生成的图表
    """
    # 提取组件分数
    components = list(signal_data['component_scores'].keys())
    scores = list(signal_data['component_scores'].values())
    
    # 计算每个组件的满分
    max_scores = {'RSI指标': 20, '价格形态': 20, '成交量分析': 15, 
                 '支撑位分析': 15, '动量指标': 15, '波动率分析': 15}
    
    # 计算百分比
    score_percentages = [scores[i] / max_scores[components[i]] * 100 for i in range(len(components))]
    
    # 添加首尾相接
    components.