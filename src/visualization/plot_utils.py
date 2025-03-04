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
    components.append(components[0])
    score_percentages.append(score_percentages[0])
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(components) - 1, endpoint=False).tolist()
    angles += [angles[0]]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # 绘制多边形
    ax.plot(angles, score_percentages, 'o-', linewidth=2, label='分数占比')
    ax.fill(angles, score_percentages, alpha=0.25)
    
    # 设置角度标签
    ax.set_thetagrids(np.degrees(angles[:-1]), components[:-1])
    
    # 设置半径标签
    ax.set_rlabel_position(0)
    ax.set_rticks([0, 25, 50, 75, 100])
    ax.set_rlim(0, 100)
    
    # 添加标题和图例
    plt.title(f"信号组成分析 (总分: {signal_data['total_score']:.1f})", size=15)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def create_plotly_chart(data, scorer, threshold=50):
    """
    使用Plotly创建交互式图表显示价格和买入信号
    
    参数:
    data (pandas.DataFrame): 价格数据
    scorer (BuySignalScorer): 信号评分器对象
    threshold (float): 信号阈值，只显示得分高于此值的信号
    
    返回:
    plotly.graph_objects.Figure: 交互式图表对象
    """
    # 计算最近所有数据点的信号强度
    signals = scorer.evaluate_recent_signals(days=len(data))
    
    # 筛选出高于阈值的信号
    strong_signals = signals[signals['total_score'] >= threshold]
    
    # 创建子图
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('价格走势与买入信号', '成交量'))
    
    # 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="K线"
        ),
        row=1, col=1
    )
    
    # 添加移动平均线
    if 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['sma_20'],
                name="20日均线",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['sma_50'],
                name="50日均线",
                line=dict(color='green', width=1)
            ),
            row=1, col=1
        )
    
    # 添加买入信号标记
    if not strong_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=strong_signals['date'],
                y=strong_signals['price'],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=12, color='red'),
                text=strong_signals['total_score'].round(1),
                textposition="top center",
                name="买入信号"
            ),
            row=1, col=1
        )
    
    # 添加成交量图
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['volume'],
            name="成交量",
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title='股票分析图表',
        xaxis_title="日期",
        yaxis_title="价格",
        height=800,
        width=1200,
        legend=dict(x=0, y=1),
        xaxis_rangeslider_visible=False
    )
    
    # 更新Y轴标题
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig


def plot_score_history(score_history, figsize=(15, 6)):
    """
    绘制信号评分历史趋势图
    
    参数:
    score_history (pandas.DataFrame): 包含日期和评分的历史数据
    figsize (tuple): 图表大小
    
    返回:
    matplotlib.figure.Figure: 生成的图表
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制评分趋势线
    ax.plot(score_history['date'], score_history['total_score'], 
            color='blue', linewidth=2, marker='o', markersize=4)
    
    # 添加阈值线
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='买入阈值')
    ax.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='强烈买入信号')
    
    # 填充区域颜色
    ax.fill_between(score_history['date'], 0, 50, color='red', alpha=0.1)
    ax.fill_between(score_history['date'], 50, 75, color='yellow', alpha=0.1)
    ax.fill_between(score_history['date'], 75, 100, color='green', alpha=0.1)
    
    # 美化图表
    ax.set_title('买入信号强度历史趋势', fontsize=15)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('信号强度', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

