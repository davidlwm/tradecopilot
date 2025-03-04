"""
英伟达(NVDA)近7天数据回测，特别关注前天的买入信号
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径，确保能导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入系统模块
from src.data.data_loader import load_from_yahoo
from src.data.data_processor import clean_price_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.examples.backtest_example import SimpleBacktester
from src.visualization.plot_utils import plot_price_with_signals, plot_signal_components

def run_nvda_backtest():
    """运行英伟达回测，关注前天买入信号"""
    print("英伟达(NVDA)买入信号回测 - 近7天数据")
    print("-" * 50)
    
    # 设置股票代码和回测参数
    symbol = "NVDA"
    period = "7d"  # 获取最近7天数据
    buy_threshold = 60  # 买入信号阈值
    sell_threshold = 40  # 卖出信号阈值
    max_position_days = 2  # 最大持仓天数（短期）
    initial_capital = 10000  # 初始资金
    
    # 计算前天日期
    today = datetime.now()
    day_before_yesterday = (today - timedelta(days=2)).date()
    
    print(f"\n1. 加载 {symbol} 股票数据，时间段: {period}...")
    try:
        # 加载数据
        data = load_from_yahoo(symbol, period=period, interval="1d")
        data = clean_price_data(data)
        
        # 显示数据范围
        start_date = data['date'].min().strftime('%Y-%m-%d')
        end_date = data['date'].max().strftime('%Y-%m-%d')
        print(f"加载了 {len(data)} 条数据，从 {start_date} 到 {end_date}")
        
        # 显示数据
        print("\n数据概览:")
        pd.set_option('display.max_columns', None)
        print(data[['date', 'open', 'high', 'low', 'close', 'volume']].head())
        
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return
    
    # 创建评分器
    print("\n2. 初始化买入信号评分系统...")
    scorer = BuySignalScorer(data)
    
    # 计算每天的评分
    print("\n3. 计算每天的买入信号评分...")
    daily_scores = []
    for i in range(len(data)):
        score_data = scorer.calculate_buy_signal_score(i)
        daily_scores.append({
            'date': score_data['date'].strftime('%Y-%m-%d'),
            'price': score_data['price'],
            'total_score': score_data['total_score'],
            'signal_strength': score_data['signal_strength'],
            'component_scores': {k: f"{v:.2f}" for k, v in score_data['component_scores'].items()}
        })
    
    # 显示每天的评分
    scores_df = pd.DataFrame(daily_scores)
    print("\n每日买入信号评分:")
    print(scores_df[['date', 'price', 'total_score', 'signal_strength']])
    
    # 特别关注前天的评分
    day_before_yesterday_str = day_before_yesterday.strftime('%Y-%m-%d')
    try:
        target_score = scores_df[scores_df['date'] == day_before_yesterday_str].iloc[0]
        print(f"\n4. 前天 ({day_before_yesterday_str}) 的买入信号详细分析:")
        print(f"   价格: ${target_score['price']:.2f}")
        print(f"   总评分: {target_score['total_score']:.2f}/100")
        print(f"   信号强度: {target_score['signal_strength']}")
        
        # 显示各组成部分评分
        print("\n   评分组成部分:")
        component_scores = eval(target_score['component_scores']) if isinstance(target_score['component_scores'], str) else target_score['component_scores']
        for component, score in component_scores.items():
            max_score = 20 if component in ['RSI指标', '价格形态'] else 15
            percentage = float(score) / max_score * 100 if isinstance(score, str) else score / max_score * 100
            print(f"   - {component}: {score}/{max_score} ({percentage:.1f}%)")
        
        # 判断前天是否有买入信号
        signal_status = "✅ 有买入信号" if target_score['total_score'] >= buy_threshold else "❌ 无买入信号"
        print(f"\n   前天买入信号状态: {signal_status}")
        
    except (IndexError, KeyError):
        print(f"\n找不到前天 ({day_before_yesterday_str}) 的数据")
    
    # 运行回测
    print("\n5. 运行回测...")
    backtester = SimpleBacktester(data, initial_capital=initial_capital)
    
    # 显示回测参数
    print("\n回测参数:")
    print(f"   - 买入信号阈值: {buy_threshold}")
    print(f"   - 卖出信号阈值: {sell_threshold}")
    print(f"   - 最大持仓天数: {max_position_days}")
    print(f"   - 初始资金: ${initial_capital}")
    
    # 运行回测
    results = backtester.run_backtest(
        scorer, 
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_position_days=max_position_days
    )
    
    # 输出回测结果
    print("\n回测结果:")
    print(f"   - 总收益率: {results['total_return_pct']:.2f}%")
    print(f"   - 交易次数: {results['total_trades']}")
    print(f"   - 胜率: {results['win_rate']:.2f}%")
    print(f"   - 平均每笔收益: {results['avg_profit_per_trade']:.2f}%")
    
    # 显示交易记录
    if backtester.trades:
        print("\n交易记录:")
        trades_df = pd.DataFrame(backtester.trades)
        # 将date列转换为字符串
        trades_df['date'] = trades_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        print(trades_df)
        
        # 查找前天的交易
        target_day_trades = trades_df[trades_df['date'] == day_before_yesterday_str]
        if not target_day_trades.empty:
            print(f"\n前天 ({day_before_yesterday_str}) 的交易:")
            print(target_day_trades)
        else:
            print(f"\n前天 ({day_before_yesterday_str}) 没有交易")
    else:
        print("\n回测期间没有交易发生")
    
    # 保存回测图表
    try:
        print("\n6. 生成回测图表...")
        fig = backtester.plot_results()
        fig.savefig(f"{symbol}_backtest_results.png")
        print(f"回测图表已保存: {symbol}_backtest_results.png")
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
    
    print("\n回测完成!")

if __name__ == "__main__":
    run_nvda_backtest()