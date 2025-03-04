"""
绘图工具模块 - 提供各种数据可视化函数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import logging
import traceback
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_price_with_signals(data, scorer=None, threshold=50, window=90, figsize=(12, 8)):
    """
    绘制价格图表并标注买入信号
    
    参数:
    data (pandas.DataFrame): 价格数据
    scorer (BuySignalScorer): 评分器对象，如果为None则不显示信号
    threshold (float): 信号阈值，高于此值的点被标记为买入信号
    window (int): 要显示的最近数据点数量
    figsize (tuple): 图表大小
    
    返回:
    matplotlib.figure.Figure: 生成的图表对象
    """
    try:
        logger.info("创建价格与信号图表")
        
        # 限制显示的数据量
        if window and len(data) > window:
            plot_data = data.iloc[-window:].copy()
        else:
            plot_data = data.copy()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # 绘制价格K线图
        date_range = pd.date_range(start=plot_data['date'].min(), end=plot_data['date'].max())
        ax1.plot(plot_data['date'], plot_data['close'], color='black', linewidth=1.5, label='收盘价')
        
        # 如果有移动平均线，也绘制
        if 'sma_20' in plot_data.columns:
            ax1.plot(plot_data['date'], plot_data['sma_20'], color='blue', linewidth=1, label='20日均线')
        if 'sma_50' in plot_data.columns:
            ax1.plot(plot_data['date'], plot_data['sma_50'], color='orange', linewidth=1, label='50日均线')
        
        # 绘制布林带
        if all(col in plot_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax1.plot(plot_data['date'], plot_data['bb_upper'], color='grey', linestyle='--', alpha=0.6)
            ax1.plot(plot_data['date'], plot_data['bb_middle'], color='grey', linestyle='-', alpha=0.6)
            ax1.plot(plot_data['date'], plot_data['bb_lower'], color='grey', linestyle='--', alpha=0.6)
            
            # 填充布林带区域
            ax1.fill_between(
                plot_data['date'], 
                plot_data['bb_upper'], 
                plot_data['bb_lower'], 
                color='grey', 
                alpha=0.1
            )
        
        # 如果提供了评分器，绘制信号标记
        signal_dates = []
        signal_prices = []
        signal_scores = []
        
        if scorer:
            # 为每个点计算评分
            for i in range(len(plot_data)):
                try:
                    score_data = scorer.calculate_buy_signal_score(len(data) - len(plot_data) + i)
                    
                    # 保存评分数据
                    plot_data.loc[plot_data.index[i], 'score'] = score_data['total_score']
                    
                    # 如果评分超过阈值，记录为信号点
                    if score_data['total_score'] >= threshold:
                        signal_dates.append(plot_data['date'].iloc[i])
                        signal_prices.append(plot_data['close'].iloc[i])
                        signal_scores.append(score_data['total_score'])
                except Exception as e:
                    logger.warning(f"计算索引 {i} 的评分时出错: {str(e)}")
            
            # 绘制信号标记
            if signal_dates:
                for date, price, score in zip(signal_dates, signal_prices, signal_scores):
                    marker_size = 100 + (score - threshold) * 2  # 分数越高，标记越大
                    ax1.scatter(date, price, s=marker_size, color='green', marker='^', zorder=5)
                    
                    # 在标记上方添加评分文本
                    ax1.annotate(
                        f"{score:.0f}", 
                        (date, price), 
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8
                    )
        
            # 绘制评分变化
            if 'score' in plot_data.columns:
                ax2.plot(plot_data['date'], plot_data['score'], color='purple', linewidth=1.5)
                ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
                ax2.fill_between(plot_data['date'], plot_data['score'], 0, where=plot_data['score'] >= threshold, color='green', alpha=0.3)
                ax2.fill_between(plot_data['date'], plot_data['score'], 0, where=plot_data['score'] < threshold, color='grey', alpha=0.2)
                
                # 绘制信号强度区域
                ax2.axhspan(80, 100, color='darkgreen', alpha=0.1)
                ax2.axhspan(70, 80, color='green', alpha=0.1)
                ax2.axhspan(60, 70, color='yellowgreen', alpha=0.1)
                ax2.axhspan(50, 60, color='yellow', alpha=0.1)
                ax2.axhspan(40, 50, color='orange', alpha=0.1)
                ax2.axhspan(30, 40, color='salmon', alpha=0.1)
                ax2.axhspan(0, 30, color='lightgrey', alpha=0.1)
                
                # 添加信号强度标签
                ax2.text(plot_data['date'].iloc[0], 90, '极强', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 75, '强', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 65, '中强', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 55, '中等', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 45, '弱', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 35, '极弱', fontsize=8, ha='left', va='center')
                ax2.text(plot_data['date'].iloc[0], 15, '无信号', fontsize=8, ha='left', va='center')
        
        # 设置坐标轴格式
        ax1.set_title('价格走势与买入信号', fontsize=15)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax2.set_title('买入信号评分', fontsize=15)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('评分', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 日期格式化
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # 调整布局
        plt.tight_layout()
        
        logger.info("价格与信号图表创建完成")
        return fig
    
    except Exception as e:
        logger.error(f"创建价格与信号图表时出错: {str(e)}")
        traceback.print_exc()
        plt.close(fig)  # 关闭可能创建的不完整图表
        raise

def plot_signal_components(score_data, figsize=(10, 8)):
    """
    使用雷达图显示买入信号的各组成部分
    
    参数:
    score_data (dict): 包含组成部分评分的字典，由BuySignalScorer.calculate_buy_signal_score()返回
    figsize (tuple): 图表大小
    
    返回:
    matplotlib.figure.Figure: 生成的图表对象
    """
    try:
        logger.info("创建信号组成雷达图")
        
        # 提取组成部分得分
        component_scores = score_data['component_scores']
        
        # 定义各组成部分的最大值
        max_scores = {
            'RSI指标': 20,
            '价格形态': 20,
            '成交量分析': 15,
            '支撑位分析': 15,
            '动量指标': 15,
            '波动率分析': 15
        }
        
        # 提取数据
        categories = list(component_scores.keys())
        values = [component_scores[category] for category in categories]
        max_values = [max_scores[category] for category in categories]
        
        # 计算评分百分比
        percentages = [value / max_value * 100 for value, max_value in zip(values, max_values)]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        
        # 设置角度（顺时针，从顶部开始）
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += [values[0]]  # 闭合雷达图
        percentages += [percentages[0]]  # 闭合雷达图
        angles += [angles[0]]  # 闭合雷达图
        categories += [categories[0]]  # 闭合雷达图
        
        # 绘制雷达图
        ax.plot(angles, percentages, color='blue', linewidth=2)
        ax.fill(angles, percentages, color='blue', alpha=0.25)
        
        # 自定义绘图
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=12)
        
        # 添加同心圆网格
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_rlim(0, 100)
        
        # 添加详细信息
        plt.figtext(0.5, 0.95, f'买入信号评分: {score_data["total_score"]:.1f}/100 ({score_data["signal_strength"]})', 
                    ha='center', fontsize=15, weight='bold')
        plt.figtext(0.5, 0.91, f'日期: {score_data["date"].strftime("%Y-%m-%d")}   价格: {score_data["price"]:.2f}', 
                    ha='center', fontsize=12)
        
        # 添加各组成部分具体分数
        for i, (category, value, max_value, percentage) in enumerate(zip(categories[:-1], values[:-1], max_values, percentages[:-1])):
            angle = angles[i]
            x = 0.05 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            plt.figtext(x, y, f'{value:.1f}/{max_value} ({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        # 调整布局
        plt.tight_layout()
        
        logger.info("信号组成雷达图创建完成")
        return fig
    
    except Exception as e:
        logger.error(f"创建信号组成雷达图时出错: {str(e)}")
        traceback.print_exc()
        plt.close(fig)  # 关闭可能创建的不完整图表
        raise

def plot_score_history(signals_df, figsize=(12, 8), threshold=50):
    """
    绘制信号得分的历史趋势图
    
    参数:
    signals_df (pandas.DataFrame): 包含日期、价格和总分的DataFrame
    figsize (tuple): 图表大小
    threshold (float): 信号阈值
    
    返回:
    matplotlib.figure.Figure: 生成的图表对象
    """
    try:
        logger.info("创建信号历史趋势图")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
        
        # 绘制价格走势
        ax1.plot(signals_df['date'], signals_df['price'], color='black', linewidth=1.5)
        ax1.set_title('价格走势', fontsize=15)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 绘制评分趋势
        ax2.plot(signals_df['date'], signals_df['total_score'], color='blue', linewidth=2)
        ax2.set_title('买入信号评分趋势', fontsize=15)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('评分', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'信号阈值 ({threshold}分)')
        ax2.grid(True, alpha=0.3)
        
        # 填充超过阈值的区域
        ax2.fill_between(signals_df['date'], signals_df['total_score'], threshold, 
                         where=signals_df['total_score'] >= threshold, 
                         color='green', alpha=0.3, label='买入信号')
        
        # 标记最高分
        max_score_idx = signals_df['total_score'].idxmax()
        max_score = signals_df['total_score'].loc[max_score_idx]
        max_date = signals_df['date'].loc[max_score_idx]
        ax2.annotate(f'最高分: {max_score:.1f}',
                    xy=(max_date, max_score),
                    xytext=(max_date, max_score + 5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
                    horizontalalignment='center',
                    fontsize=10)
        
        # 绘制信号强度区域
        ax2.axhspan(80, 100, color='darkgreen', alpha=0.1, label='极强信号 (80-100)')
        ax2.axhspan(70, 80, color='green', alpha=0.1, label='强信号 (70-79)')
        ax2.axhspan(60, 70, color='yellowgreen', alpha=0.1, label='中强信号 (60-69)')
        ax2.axhspan(50, 60, color='yellow', alpha=0.1, label='中等信号 (50-59)')
        ax2.axhspan(40, 50, color='orange', alpha=0.1, label='弱信号 (40-49)')
        ax2.axhspan(30, 40, color='salmon', alpha=0.1, label='极弱信号 (30-39)')
        ax2.axhspan(0, 30, color='lightgrey', alpha=0.1, label='无信号 (0-29)')
        
        # 添加图例
        ax2.legend(loc='upper left', fontsize=8)
        
        # 日期格式化
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # 添加统计信息
        above_threshold = signals_df[signals_df['total_score'] >= threshold]
        plt.figtext(0.02, 0.02, 
                   f'统计: 总天数={len(signals_df)}  '
                   f'信号天数={len(above_threshold)} ({len(above_threshold)/len(signals_df)*100:.1f}%)  '
                   f'最高分={max_score:.1f}  '
                   f'平均分={signals_df["total_score"].mean():.1f}', 
                   fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        logger.info("信号历史趋势图创建完成")
        return fig
    
    except Exception as e:
        logger.error(f"创建信号历史趋势图时出错: {str(e)}")
        traceback.print_exc()
        plt.close(fig)  # 关闭可能创建的不完整图表
        raise

def create_plotly_chart(data, scorer=None, threshold=50, window=90):
    """
    创建交互式Plotly图表
    
    参数:
    data (pandas.DataFrame): 价格数据
    scorer (BuySignalScorer): 评分器对象，如果为None则不显示信号
    threshold (float): 信号阈值，高于此值的点被标记为买入信号
    window (int): 要显示的最近数据点数量
    
    返回:
    plotly.graph_objects.Figure: 生成的Plotly图表对象
    """
    try:
        logger.info("创建交互式Plotly图表")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("未安装plotly库，请使用pip install plotly安装")
            raise ImportError("需要plotly库。请使用pip install plotly安装。")
        
        # 限制显示的数据量
        if window and len(data) > window:
            plot_data = data.iloc[-window:].copy()
        else:
            plot_data = data.copy()
        
        # 创建子图布局
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.7, 0.3],
                            subplot_titles=('价格走势与买入信号', '买入信号评分'))
        
        # 绘制K线图
        fig.add_trace(go.Candlestick(
            x=plot_data['date'],
            open=plot_data['open'], 
            high=plot_data['high'],
            low=plot_data['low'], 
            close=plot_data['close'],
            name='K线'
        ), row=1, col=1)
        
        # 添加移动平均线
        if 'sma_20' in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=plot_data['date'], 
                y=plot_data['sma_20'],
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0.7)', width=1),
                name='20日均线'
            ), row=1, col=1)
        
        if 'sma_50' in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=plot_data['date'], 
                y=plot_data['sma_50'],
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1),
                name='50日均线'
            ), row=1, col=1)
        
        # 添加布林带
        if all(col in plot_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(go.Scatter(
                x=plot_data['date'], 
                y=plot_data['bb_upper'],
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.4)', width=1, dash='dash'),
                name='布林带上轨'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_data['date'], 
                y=plot_data['bb_middle'],
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.4)', width=1),
                name='布林带中轨'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_data['date'], 
                y=plot_data['bb_lower'],
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.4)', width=1, dash='dash'),
                name='布林带下轨'
            ), row=1, col=1)
            
            # 添加布林带填充区域
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['bb_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='布林带区域'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['bb_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False,
                name='布林带区域'
            ), row=1, col=1)
        
        # 如果提供了评分器，添加信号标记
        signal_dates = []
        signal_prices = []
        signal_scores = []
        
        if scorer:
            # 为每个点计算评分
            for i in range(len(plot_data)):
                try:
                    score_data = scorer.calculate_buy_signal_score(len(data) - len(plot_data) + i)
                    
                    # 保存评分数据
                    plot_data.loc[plot_data.index[i], 'score'] = score_data['total_score']
                    
                    # 如果评分超过阈值，记录为信号点
                    if score_data['total_score'] >= threshold:
                        signal_dates.append(plot_data['date'].iloc[i])
                        signal_prices.append(plot_data['close'].iloc[i])
                        signal_scores.append(score_data['total_score'])
                except Exception as e:
                    logger.warning(f"计算索引 {i} 的评分时出错: {str(e)}")
            
            # 添加信号标记
            if signal_dates:
                fig.add_trace(go.Scatter(
                    x=signal_dates,
                    y=signal_prices,
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    ),
                    text=[f"{s:.0f}" for s in signal_scores],
                    textposition="top center",
                    name='买入信号'
                ), row=1, col=1)
            
            # 添加评分图
            if 'score' in plot_data.columns:
                fig.add_trace(go.Scatter(
                    x=plot_data['date'],
                    y=plot_data['score'],
                    mode='lines',
                    line=dict(color='purple', width=2),
                    name='买入信号评分'
                ), row=2, col=1)
                
                # 添加评分阈值线
                fig.add_shape(
                    type="line",
                    x0=plot_data['date'].iloc[0],
                    y0=threshold,
                    x1=plot_data['date'].iloc[-1],
                    y1=threshold,
                    line=dict(color="red", width=2, dash="dash"),
                    row=2, col=1
                )
                
                # 强信号区域
                fig.add_trace(go.Scatter(
                    x=plot_data['date'],
                    y=[80] * len(plot_data),
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=plot_data['date'],
                    y=[100] * len(plot_data),
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 0, 0.1)',
                    showlegend=False,
                    name='极强信号区域'
                ), row=2, col=1)
                
                # 添加信号强度区域标签
                fig.add_annotation(
                    x=plot_data['date'].iloc[0],
                    y=90,
                    text="极强",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="left",
                    yanchor="middle",
                    row=2, col=1
                )
                
                fig.add_annotation(
                    x=plot_data['date'].iloc[0],
                    y=75,
                    text="强",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="left",
                    yanchor="middle",
                    row=2, col=1
                )
                
                fig.add_annotation(
                    x=plot_data['date'].iloc[0],
                    y=65,
                    text="中强",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="left",
                    yanchor="middle",
                    row=2, col=1
                )
                
                fig.add_annotation(
                    x=plot_data['date'].iloc[0],
                    y=55,
                    text="中等",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="left",
                    yanchor="middle",
                    row=2, col=1
                )
        
        # 配置布局
        fig.update_layout(
            title='多因子买入信号评分图表',
            xaxis_rangeslider_visible=False,
            yaxis_title='价格',
            yaxis2_title='评分',
            xaxis2_title='日期',
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis2_range=[0, 100]
        )
        
        # 配置X轴
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # 隐藏周末
            ]
        )
        
        logger.info("交互式Plotly图表创建完成")
        return fig
    
    except Exception as e:
        logger.error(f"创建交互式Plotly图表时出错: {str(e)}")
        traceback.print_exc()
        raise