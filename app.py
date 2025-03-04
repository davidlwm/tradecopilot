"""
多因子买入信号评分系统 - Streamlit界面

为现有多因子买入信号评分系统提供Web界面
集成现有的基础功能、回测功能和实时监控功能
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目已有模块
from src.data.data_loader import load_from_yahoo, generate_sample_data
from src.data.data_processor import clean_price_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.visualization.plot_utils import plot_price_with_signals, plot_signal_components, plot_score_history
from src.examples.backtest_example import SimpleBacktester
from src.examples.realtime_scoring import RealTimeMonitor

# 配置页面
st.set_page_config(
    page_title="多因子买入信号评分系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 缓存数据加载函数
@st.cache_data(ttl=3600)
def load_stock_data(symbol, period, interval="1d"):
    """缓存数据加载结果，减少重复请求"""
    try:
        data = load_from_yahoo(symbol, period=period, interval=interval)
        data = clean_price_data(data)
        return data, None
    except Exception as e:
        return None, str(e)

def single_stock_analysis():
    """单股票基本分析功能"""
    st.header("股票买入信号分析")
    
    # 输入参数
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("股票代码", "AAPL").strip().upper()
    with col2:
        period = st.selectbox("时间跨度", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        threshold = st.slider("买入信号阈值", 40, 80, 60)
        
    # 分析按钮
    if st.button("分析"):
        if not symbol:
            st.error("请输入股票代码")
            return
            
        # 加载数据
        with st.spinner(f"正在加载 {symbol} 数据..."):
            data, error = load_stock_data(symbol, period)
            
        if error:
            st.error(f"加载数据失败: {error}")
            return
        if data is None or len(data) == 0:
            st.error("未找到数据或数据为空")
            return
            
        st.success(f"成功加载 {len(data)} 条数据记录，从 {data['date'].iloc[0].strftime('%Y-%m-%d')} 到 {data['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        # 创建评分器
        with st.spinner("计算买入信号评分..."):
            scorer = BuySignalScorer(data)
            score_data = scorer.calculate_buy_signal_score()
        
        # 显示当前评分
        st.subheader("当前买入信号评分")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总评分", f"{score_data['total_score']:.2f}/100")
        with col2:
            st.metric("信号强度", score_data['signal_strength'])
        with col3:
            st.metric("当前价格", f"${score_data['price']:.2f}")
        
        # 展示各组件评分
        st.subheader("评分组成部分")
        component_data = []
        for component, score in score_data['component_scores'].items():
            max_score = 20 if component in ['RSI指标', '价格形态'] else 15
            percentage = score / max_score * 100
            component_data.append({
                "组件": component,
                "得分": f"{score:.2f}/{max_score}",
                "百分比": f"{percentage:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(component_data), use_container_width=True)
        
        # 可视化图表
        tab1, tab2, tab3 = st.tabs(["价格与信号图", "评分雷达图", "评分历史"])
        
        with tab1:
            with st.spinner("生成价格与信号图..."):
                price_fig = plot_price_with_signals(data, scorer, threshold=threshold)
                st.pyplot(price_fig)
        
        with tab2:
            with st.spinner("生成评分雷达图..."):
                radar_fig = plot_signal_components(score_data)
                st.pyplot(radar_fig)
                
        with tab3:
            with st.spinner("计算历史评分..."):
                # 获取最近30天的历史评分
                days_to_evaluate = min(30, len(data))
                recent_signals = scorer.evaluate_recent_signals(days=days_to_evaluate)
                
                history_fig = plot_score_history(recent_signals)
                st.pyplot(history_fig)
                
                # 显示最近的买入信号
                buy_signals = recent_signals[recent_signals['total_score'] >= threshold]
                if not buy_signals.empty:
                    st.subheader(f"最近 {days_to_evaluate} 天内的买入信号")
                    for _, signal in buy_signals.iterrows():
                        st.write(f"📅 {signal['date'].strftime('%Y-%m-%d')} - 评分: {signal['total_score']:.1f} - {signal['signal_strength']}")
                else:
                    st.info(f"最近 {days_to_evaluate} 天内没有评分超过 {threshold} 的买入信号")

def run_backtest():
    """运行回测功能"""
    st.header("买入信号策略回测")
    
    # 回测设置
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("股票代码", "AAPL").strip().upper()
        period = st.selectbox("回测周期", ["1y", "2y", "3y", "5y"], index=1)
    
    with col2:
        initial_capital = st.number_input("初始资金", 1000, 1000000, 10000, step=1000)
        buy_threshold = st.slider("买入信号阈值", 40, 80, 60)
        sell_threshold = st.slider("卖出信号阈值", 20, 60, 40)
        max_position_days = st.number_input("最大持仓天数", 5, 60, 20)
    
    # 开始回测
    if st.button("开始回测"):
        if not symbol:
            st.error("请输入股票代码")
            return
            
        # 加载数据
        with st.spinner(f"正在加载 {symbol} 数据..."):
            data, error = load_stock_data(symbol, period)
            
        if error:
            st.error(f"加载数据失败: {error}")
            return
        if data is None or len(data) == 0:
            st.error("未找到数据或数据为空")
            return
            
        st.success(f"成功加载 {len(data)} 条数据记录，从 {data['date'].iloc[0].strftime('%Y-%m-%d')} 到 {data['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        # 创建评分器和回测器
        with st.spinner("运行回测..."):
            # 创建评分器
            scorer = BuySignalScorer(data)
            
            # 创建回测器
            backtester = SimpleBacktester(data, initial_capital=initial_capital)
            
            # 运行回测
            results = backtester.run_backtest(
                scorer, 
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                max_position_days=max_position_days
            )
        
        # 显示回测结果
        st.subheader("回测结果")
        
        # 结果指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总收益率", f"{results['total_return_pct']:.2f}%")
        with col2:
            st.metric("年化收益率", f"{results['annualized_return']:.2f}%")
        with col3:
            st.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
        with col4:
            st.metric("胜率", f"{results['win_rate']:.2f}%")
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("交易次数", f"{results['total_trades']}")
        with col2:
            st.metric("平均持仓天数", f"{results['avg_holding_period']:.2f}")
        with col3:
            st.metric("平均每笔收益", f"{results['avg_profit_per_trade']:.2f}%")
        with col4:
            st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
        
        # 回测图表
        try:
            with st.spinner("生成回测图表..."):
                fig = backtester.plot_results()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"生成回测图表时出错: {str(e)}")
        
        # 交易记录
        if backtester.trades:
            st.subheader("交易记录")
            trades_df = pd.DataFrame(backtester.trades)
            st.dataframe(trades_df)
            
            # 下载按钮
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "下载交易记录",
                csv,
                f"{symbol}_trades.csv",
                "text/csv",
                key="download-trades"
            )

def run_multi_stock_monitor():
    """运行多股票监控功能"""
    st.header("多股票买入信号监控")
    
    # 输入股票列表
    symbols_text = st.text_area(
        "输入要监控的股票代码 (每行一个)",
        "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
        height=100
    )
    
    symbols = [s.strip().upper() for s in symbols_text.split("\n") if s.strip()]
    
    # 设置参数
    col1, col2, col3 = st.columns(3)
    with col1:
        days_lookback = st.slider("历史数据天数", 30, 365, 180)
    with col2:
        threshold = st.slider("买入信号阈值", 40, 80, 60)
    with col3:
        use_mock = st.checkbox("使用模拟数据 (用于演示)", True)
    
    # 开始监控
    if st.button("开始监控"):
        if not symbols:
            st.error("请至少输入一个股票代码")
            return
            
        st.success(f"开始监控 {len(symbols)} 只股票")
        
        # 创建监控器
        monitor = RealTimeMonitor(symbols, threshold=threshold)
        
        # 初始化监控器
        with st.spinner("初始化数据..."):
            try:
                monitor.initialize(days_lookback=days_lookback)
                st.success("数据初始化完成")
            except Exception as e:
                st.error(f"初始化数据时出错: {str(e)}")
                # 使用模拟数据
                st.warning("使用模拟数据继续...")
                for symbol in symbols:
                    if symbol not in monitor.data:
                        data = generate_sample_data(days=252)
                        monitor.data[symbol] = data
                        monitor.scorers[symbol] = BuySignalScorer(data)
                        monitor.latest_scores[symbol] = monitor.scorers[symbol].calculate_buy_signal_score()
        
        # 显示初始评分
        monitor.check_signals()
        
        # 创建可刷新的容器
        monitor_container = st.empty()
        
        # 设置最大迭代次数
        max_iterations = 5 if use_mock else 1
        
        # 运行监控循环
        for i in range(max_iterations):
            if i > 0:
                with st.spinner(f"更新数据 (第 {i+1} 次)..."):
                    monitor.update_data(use_mock=use_mock)
                    
                with monitor_container.container():
                    # 显示当前时间
                    st.write(f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 显示所有股票的评分
                    scores = []
                    for symbol, score_data in monitor.latest_scores.items():
                        scores.append({
                            "股票": symbol,
                            "价格": score_data['price'],
                            "评分": score_data['total_score'],
                            "信号强度": score_data['signal_strength'],
                            "日期": score_data['date'].strftime('%Y-%m-%d')
                        })
                    
                    # 按评分排序
                    scores_df = pd.DataFrame(scores).sort_values("评分", ascending=False)
                    st.dataframe(scores_df, use_container_width=True)
                    
                    # 显示高评分股票
                    high_scores = scores_df[scores_df["评分"] >= threshold]
                    if not high_scores.empty:
                        st.subheader(f"发现 {len(high_scores)} 只股票产生买入信号")
                        
                        for _, stock in high_scores.iterrows():
                            symbol = stock["股票"]
                            with st.expander(f"{symbol} - 评分: {stock['评分']:.2f} - {stock['信号强度']}"):
                                score_data = monitor.latest_scores[symbol]
                                
                                # 显示组成部分
                                st.write("**评分组成部分:**")
                                for component, score in score_data['component_scores'].items():
                                    max_score = 20 if component in ['RSI指标', '价格形态'] else 15
                                    st.write(f"- {component}: {score:.2f}/{max_score} ({score/max_score*100:.1f}%)")
                                
                                # 显示图表
                                try:
                                    if symbol in monitor.scorers:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            price_fig = plot_price_with_signals(
                                                monitor.data[symbol],
                                                monitor.scorers[symbol],
                                                threshold=threshold
                                            )
                                            st.pyplot(price_fig)
                                        
                                        with col2:
                                            radar_fig = plot_signal_components(score_data)
                                            st.pyplot(radar_fig)
                                except Exception as e:
                                    st.error(f"生成图表时出错: {str(e)}")
                    else:
                        st.info(f"没有股票的评分超过阈值 {threshold}")
            
            # 等待3秒后更新 (在实际应用中可以设置更长的间隔)
            if i < max_iterations - 1:
                time.sleep(3)

def main():
    """主函数"""
    st.title("多因子买入信号评分系统")
    
    # 侧边栏导航
    st.sidebar.title("导航")
    app_mode = st.sidebar.radio(
        "选择功能", 
        ["单支股票分析", "多股票监控", "策略回测", "关于系统"]
    )
    
    # 显示选定的功能
    if app_mode == "单支股票分析":
        single_stock_analysis()
        
    elif app_mode == "多股票监控":
        run_multi_stock_monitor()
        
    elif app_mode == "策略回测":
        run_backtest()
        
    elif app_mode == "关于系统":
        st.header("关于多因子买入信号评分系统")
        
        st.write("""
        ## 系统介绍
        
        多因子买入信号评分系统是一个基于技术分析的股票买入信号识别工具。它通过分析多个技术指标，
        为潜在的股票买入时机提供量化评分，帮助投资者更客观地评估市场机会。
        
        ### 评分系统原理
        
        买入信号评分基于六个维度的分析，每个维度权重可配置：
        
        1. **RSI指标** (20分): 评估价格是否处于超卖区域
        2. **价格形态** (20分): 分析布林带、移动平均线和蜡烛图形态
        3. **成交量分析** (15分): 评估相对成交量和OBV指标
        4. **支撑位分析** (15分): 检测价格是否接近支撑位或重要斐波那契回调位
        5. **动量指标** (15分): 分析随机指标、威廉指标和MACD
        6. **波动率分析** (15分): 评估ATR和布林带宽度变化
        
        系统根据总分(0-100)给出信号强度评估，从"无买入信号"到"极强买入信号"。
        
        ### 使用注意事项
        
        - 本系统仅提供技术分析参考，不构成投资建议
        - 建议将技术分析结果与基本面分析相结合
        - 在实际交易中设置适当的止损策略
        - 回测结果只是历史表现，不代表未来收益
        """)
        
        st.info("本系统数据来源于Yahoo Finance，仅用于研究和学习目的。")

if __name__ == "__main__":
    main()