"""
多因子买入信号评分系统 - 回测示例

演示如何使用评分系统进行历史回测和策略评估
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需模块
from src.data.data_loader import load_from_yahoo, generate_sample_data
from src.data.data_processor import clean_price_data
from src.scorer.buy_signal_scorer import BuySignalScorer

class SimpleBacktester:
    """简单的回测系统"""
    
    def __init__(self, data, initial_capital=10000):
        """
        初始化回测系统
        
        参数:
        data (pandas.DataFrame): 价格数据
        initial_capital (float): 初始资金
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, scorer, buy_threshold=60, sell_threshold=40, max_position_days=20):
        """
        运行回测
        
        参数:
        scorer (BuySignalScorer): 评分系统
        buy_threshold (float): 买入阈值
        sell_threshold (float): 卖出阈值
        max_position_days (int): 最大持仓天数
        
        返回:
        dict: 回测结果
        """
        # 重置状态
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        
        # 记录每日权益
        self.equity_curve = []
        
        # 持仓计数器
        days_in_position = 0
        entry_price = 0
        
        # 遍历数据点
        for i in range(20, len(self.data)):  # 跳过前20个点，因为技术指标需要历史数据
            current_date = self.data['date'].iloc[i]
            current_price = self.data['close'].iloc[i]
            
            # 计算当前的评分
            score_data = scorer.calculate_buy_signal_score(i)
            score = score_data['total_score']
            
            # 记录当前权益
            current_equity = self.capital
            if self.position > 0:
                current_equity += self.position * current_price
            
            self.equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'price': current_price,
                'position': self.position,
                'score': score
            })
            
            # 交易逻辑
            if self.position == 0:  # 无持仓时，考虑买入
                if score >= buy_threshold:
                    # 计算可买入的股数
                    shares_to_buy = int(self.capital / current_price)
                    if shares_to_buy > 0:
                        self.position = shares_to_buy
                        self.capital -= shares_to_buy * current_price
                        entry_price = current_price
                        days_in_position = 0
                        
                        # 记录交易
                        self.trades.append({
                            'type': 'buy',
                            'date': current_date,
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': shares_to_buy * current_price,
                            'score': score
                        })
            
            else:  # 有持仓时，考虑卖出
                days_in_position += 1
                
                # 卖出条件：信号弱于卖出阈值 或 持仓时间超过最大天数 或 止损(亏损>5%)
                if (score <= sell_threshold or 
                    days_in_position >= max_position_days or 
                    current_price < entry_price * 0.95):
                    
                    # 卖出所有持仓
                    sell_value = self.position * current_price
                    self.capital += sell_value
                    
                    # 记录交易
                    self.trades.append({
                        'type': 'sell',
                        'date': current_date,
                        'price': current_price,
                        'shares': self.position,
                        'value': sell_value,
                        'score': score,
                        'days_held': days_in_position,
                        'profit_pct': (current_price / entry_price - 1) * 100
                    })
                    
                    # 重置持仓
                    self.position = 0
                    days_in_position = 0
                    entry_price = 0
        
        # 清仓最后的持仓(如果有)
        if self.position > 0:
            last_price = self.data['close'].iloc[-1]
            sell_value = self.position * last_price
            self.capital += sell_value
            
            # 记录交易
            self.trades.append({
                'type': 'sell',
                'date': self.data['date'].iloc[-1],
                'price': last_price,
                'shares': self.position,
                'value': sell_value,
                'score': scorer.calculate_buy_signal_score(-1)['total_score'],
                'days_held': days_in_position,
                'profit_pct': (last_price / entry_price - 1) * 100
            })
            
            self.position = 0
        
        # 计算回测结果统计
        results = self._calculate_stats()
        
        return results
    
    def _calculate_stats(self):
        """计算回测统计结果"""
        if not self.trades or len(self.trades) < 2:
            return {
                'total_return_pct': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_holding_period': 0,
                'avg_profit_per_trade': 0,
                'sharpe_ratio': 0
            }
        
        # 将交易列表转换为DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # 将权益曲线转换为DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 计算总收益率
        final_equity = self.capital
        total_return = final_equity / self.initial_capital - 1
        
        # 计算年化收益率
        start_date = self.data['date'].iloc[0]
        end_date = self.data['date'].iloc[-1]
        years = (end_date - start_date).days / 365
        annualized_return = (1 + total_return) ** (1 / max(years, 0.1)) - 1
        
        # 计算最大回撤
        equity_df['equity_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak']
        max_drawdown = -equity_df['drawdown'].min()
        
        # 过滤出卖出交易，计算胜率和平均盈利
        sell_trades = trades_df[trades_df['type'] == 'sell']
        profits = sell_trades['profit_pct'].values
        
        win_rate = len(profits[profits > 0]) / len(profits) if len(profits) > 0 else 0
        avg_profit = profits.mean() if len(profits) > 0 else 0
        avg_holding_period = sell_trades['days_held'].mean() if len(sell_trades) > 0 else 0
        
        # 计算夏普比率 (简化版)
        daily_returns = equity_df['equity'].pct_change().dropna()
        risk_free_rate = 0.02 / 252  # 假设年化无风险利率2%
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / (daily_returns.std() + 1e-10) * np.sqrt(252)
        
        # 组合结果
        results = {
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return * 100,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'total_trades': len(sell_trades),
            'avg_holding_period': avg_holding_period,
            'avg_profit_per_trade': avg_profit,
            'sharpe_ratio': sharpe_ratio
        }
        
        return results
    
    def plot_results(self):
        """绘制回测结果图表"""
        if not self.equity_curve:
            print("没有回测数据可以绘制")
            return
        
        # 创建DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex=True)
        
        # 绘制权益曲线
        ax1.plot(equity_df['date'], equity_df['equity'], label='账户权益')
        ax1.set_title('回测结果: 账户权益曲线', fontsize=15)
        ax1.set_ylabel('账户价值', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记交易点
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            
            # 获取权益曲线对应日期的权益值
            if not buy_trades.empty:
                buy_equities = []
                for date in buy_trades['date']:
                    matches = equity_df[equity_df['date'] == date]
                    if not matches.empty:
                        buy_equities.append(matches['equity'].values[0])
                
                # 绘制买入点
                if len(buy_equities) == len(buy_trades):
                    ax1.scatter(buy_trades['date'], buy_equities, 
                               color='green', marker='^', s=100, zorder=5, label='买入')
            
            # 绘制卖出点
            if not sell_trades.empty:
                sell_equities = []
                for date in sell_trades['date']:
                    matches = equity_df[equity_df['date'] == date]
                    if not matches.empty:
                        sell_equities.append(matches['equity'].values[0])
                
                if len(sell_equities) == len(sell_trades):
                    ax1.scatter(sell_trades['date'], sell_equities, 
                               color='red', marker='v', s=100, zorder=5, label='卖出')
        
        # 绘制价格曲线
        ax2.plot(equity_df['date'], equity_df['price'], color='blue')
        ax2.set_title('价格走势', fontsize=15)
        ax2.set_ylabel('价格', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 标记买入卖出点
        if not trades_df.empty:
            if not buy_trades.empty:
                ax2.scatter(buy_trades['date'], buy_trades['price'], 
                           color='green', marker='^', s=100, zorder=5, label='买入')
            if not sell_trades.empty:
                ax2.scatter(sell_trades['date'], sell_trades['price'], 
                           color='red', marker='v', s=100, zorder=5, label='卖出')
            ax2.legend()
        
        # 绘制评分走势
        ax3.plot(equity_df['date'], equity_df['score'], color='purple')
        ax3.set_title('信号评分走势', fontsize=15)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_ylabel('评分', fontsize=12)
        ax3.axhline(y=60, color='g', linestyle='--', alpha=0.7, label='买入阈值')
        ax3.axhline(y=40, color='r', linestyle='--', alpha=0.7, label='卖出阈值')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        return fig

def main():
    """运行回测示例"""
    print("多因子买入信号评分系统 - 回测示例")
    print("-" * 50)
    
    # 选择回测标的和时间段
    symbol = "AAPL"  # 苹果公司
    period = "5y"    # 5年数据
    
    print(f"\n1. 从Yahoo Finance加载{symbol}股票数据, 时间段:{period}...")
    try:
        data = load_from_yahoo(symbol, period=period, interval="1d")
        data = clean_price_data(data)
        print(f"加载了 {len(data)} 条数据，从 {data['date'].iloc[0].strftime('%Y-%m-%d')} 到 {data['date'].iloc[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        print("生成样本数据用于演示...")
        data = generate_sample_data(days=1000)
        print(f"生成了 {len(data)} 条样本数据")
    
    # 创建评分器
    print("\n2. 初始化买入信号评分系统...")
    scorer = BuySignalScorer(data)
    
    # 创建回测器
    print("\n3. 运行回测...")
    backtester = SimpleBacktester(data, initial_capital=10000)
    
    # 参数设置
    print("\n3.1 回测参数:")
    buy_threshold = 60   # 买入阈值
    sell_threshold = 40  # 卖出阈值
    max_position_days = 20  # 最大持仓天数
    
    print(f"  - 买入信号阈值: {buy_threshold}")
    print(f"  - 卖出信号阈值: {sell_threshold}")
    print(f"  - 最大持仓天数: {max_position_days}")
    print(f"  - 初始资金: {backtester.initial_capital}")
    
    # 运行回测
    results = backtester.run_backtest(
        scorer, 
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_position_days=max_position_days
    )
    
    # 输出回测结果
    print("\n4. 回测结果:")
    print(f"  - 总收益率: {results['total_return_pct']:.2f}%")
    print(f"  - 年化收益率: {results['annualized_return']:.2f}%")
    print(f"  - 最大回撤: {results['max_drawdown']:.2f}%")
    print(f"  - 胜率: {results['win_rate']:.2f}%")
    print(f"  - 交易次数: {results['total_trades']}")
    print(f"  - 平均持仓天数: {results['avg_holding_period']:.2f}")
    print(f"  - 平均每笔收益: {results['avg_profit_per_trade']:.2f}%")
    print(f"  - 夏普比率: {results['sharpe_ratio']:.2f}")
    
    # 绘制回测结果图表
    print("\n5. 生成回测图表...")
    try:
        fig = backtester.plot_results()
        fig.savefig("backtest_results.png")
        print("已保存回测结果图表: backtest_results.png")
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
    
    print("\n回测示例运行完成!")

if __name__ == "__main__":
    main()