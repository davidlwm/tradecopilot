"""
多因子买入信号评分系统 - 实时评分示例

模拟实时监控多只股票的买入信号强度，可用于定期运行的自动化脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入所需模块
from src.data.data_loader import load_from_yahoo, generate_sample_data
from src.data.data_processor import clean_price_data
from src.scorer.buy_signal_scorer import BuySignalScorer
from src.visualization.plot_utils import plot_price_with_signals, plot_signal_components

class RealTimeMonitor:
    """模拟实时监控多只股票的买入信号"""
    
    def __init__(self, symbols, threshold=60):
        """
        初始化监控器
        
        参数:
        symbols (list): 股票代码列表
        threshold (float): 发出提醒的信号阈值
        """
        self.symbols = symbols
        self.threshold = threshold
        self.scorers = {}
        self.latest_scores = {}
        self.data = {}
        
    def initialize(self, days_lookback=365):
        """
        初始化数据和评分器
        
        参数:
        days_lookback (int): 回溯的历史数据天数
        """
        print(f"正在初始化 {len(self.symbols)} 只股票的数据...")
        
        for symbol in self.symbols:
            print(f"加载 {symbol} 历史数据...")
            try:
                # 加载历史数据
                data = load_from_yahoo(symbol, period=f"{days_lookback}d", interval="1d")
                data = clean_price_data(data)
                
                # 保存数据
                self.data[symbol] = data
                
                # 创建评分器
                scorer = BuySignalScorer(data)
                
                # 存储评分器
                self.scorers[symbol] = scorer
                
                # 计算最新评分
                latest_score = scorer.calculate_buy_signal_score()
                self.latest_scores[symbol] = latest_score
                
                print(f"  {symbol} 初始化完成，最新评分: {latest_score['total_score']:.2f}")
                
            except Exception as e:
                print(f"  {symbol} 初始化失败: {str(e)}")
        
        print("所有股票初始化完成")
    
    def update_data(self, use_mock=False):
        """
        更新所有股票的数据和评分
        
        参数:
        use_mock (bool): 是否使用模拟数据（用于演示）
        """
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 更新数据...")
        
        for symbol in self.symbols:
            try:
                print(f"更新 {symbol} 数据...")
                
                if use_mock:
                    # 模拟数据更新（仅用于演示）
                    if symbol in self.data:
                        # 复制最后一天的数据并稍作修改
                        last_day = self.data[symbol].iloc[-1].copy()
                        new_day = pd.DataFrame([last_day])
                        
                        # 修改日期和价格
                        new_day['date'] = last_day['date'] + timedelta(days=1)
                        change_pct = np.random.normal(0, 0.015)  # 生成随机价格变化
                        new_day['close'] = last_day['close'] * (1 + change_pct)
                        new_day['open'] = last_day['close']
                        new_day['high'] = max(new_day['open'].iloc[0], new_day['close'].iloc[0]) * (1 + abs(np.random.normal(0, 0.005)))
                        new_day['low'] = min(new_day['open'].iloc[0], new_day['close'].iloc[0]) * (1 - abs(np.random.normal(0, 0.005)))
                        new_day['volume'] = last_day['volume'] * np.random.uniform(0.7, 1.3)
                        
                        # 添加新数据
                        self.data[symbol] = pd.concat([self.data[symbol], new_day]).reset_index(drop=True)
                        
                        print(f"  添加模拟数据: 日期={new_day['date'].iloc[0].strftime('%Y-%m-%d')}, 收盘价={new_day['close'].iloc[0]:.2f}")
                else:
                    # 实际更新数据（真实场景使用）
                    new_data = load_from_yahoo(symbol, period="7d", interval="1d")
                    new_data = clean_price_data(new_data)
                    
                    if symbol in self.data:
                        # 获取最后一个日期
                        last_date = self.data[symbol]['date'].iloc[-1]
                        
                        # 过滤出新数据
                        new_data = new_data[new_data['date'] > last_date]
                        
                        if not new_data.empty:
                            # 添加新数据
                            self.data[symbol] = pd.concat([self.data[symbol], new_data]).reset_index(drop=True)
                            print(f"  添加了 {len(new_data)} 条新数据")
                        else:
                            print("  没有新数据")
                    else:
                        self.data[symbol] = new_data
                        print(f"  加载了 {len(new_data)} 条数据")
                
                # 更新评分器
                if symbol in self.data:
                    # 重新创建评分器
                    self.scorers[symbol] = BuySignalScorer(self.data[symbol])
                    
                    # 计算最新评分
                    latest_score = self.scorers[symbol].calculate_buy_signal_score()
                    self.latest_scores[symbol] = latest_score
                    
                    print(f"  {symbol} 更新完成，最新评分: {latest_score['total_score']:.2f}")
                
            except Exception as e:
                print(f"  {symbol} 更新失败: {str(e)}")
    
    def check_signals(self):
        """检查是否有股票达到买入信号阈值"""
        print("\n检查买入信号...")
        
        # 按评分排序
        sorted_scores = sorted(
            self.latest_scores.items(), 
            key=lambda x: x[1]['total_score'], 
            reverse=True
        )
        
        # 显示所有股票的评分
        print("\n当前评分排名:")
        print("-" * 65)
        print(f"{'排名':^5}|{'股票':^10}|{'评分':^8}|{'信号强度':^15}|{'价格':^10}|{'日期':^12}")
        print("-" * 65)
        
        for i, (symbol, score_data) in enumerate(sorted_scores):
            print(f"{i+1:^5}|{symbol:^10}|{score_data['total_score']:^8.2f}|"
                  f"{score_data['signal_strength']:^15}|{score_data['price']:^10.2f}|"
                  f"{score_data['date'].strftime('%Y-%m-%d'):^12}")
        
        print("-" * 65)
        
        # 筛选出高于阈值的股票
        strong_signals = [(symbol, score_data) for symbol, score_data in sorted_scores 
                         if score_data['total_score'] >= self.threshold]
        
        # 显示买入信号提醒
        if strong_signals:
            print(f"\n发现 {len(strong_signals)} 只股票产生强买入信号 (评分 >= {self.threshold}):")
            
            for symbol, score_data in strong_signals:
                print(f"\n股票: {symbol}")
                print(f"评分: {score_data['total_score']:.2f}/100")
                print(f"信号强度: {score_data['signal_strength']}")
                print(f"价格: {score_data['price']:.2f}")
                print(f"日期: {score_data['date'].strftime('%Y-%m-%d')}")
                
                print("\n各组成部分评分:")
                for component, score in score_data['component_scores'].items():
                    max_score = 20 if component in ['RSI指标', '价格形态'] else 15
                    print(f"  - {component}: {score:.2f}/{max_score} ({score/max_score*100:.1f}%)")
                
                # 生成并保存图表
                if symbol in self.scorers:
                    self._generate_charts(symbol)
                
        else:
            print(f"\n没有股票产生强买入信号 (评分 >= {self.threshold})")
    
    def _generate_charts(self, symbol):
        """生成股票的图表"""
        try:
            scorer = self.scorers[symbol]
            
            # 绘制价格图表
            price_fig = plot_price_with_signals(scorer.data, scorer, threshold=self.threshold)
            price_fig.savefig(f"{symbol}_price_signals.png")
            
            # 绘制最新信号雷达图
            radar_fig = plot_signal_components(self.latest_scores[symbol])
            radar_fig.savefig(f"{symbol}_signal_components.png")
            
            print(f"已保存 {symbol} 图表")
            plt.close('all')  # 关闭所有图表，释放内存
            
        except Exception as e:
            print(f"生成 {symbol} 图表失败: {str(e)}")
    
    def run_monitor(self, update_interval=3600, max_iterations=5, use_mock=True):
        """
        运行监控循环
        
        参数:
        update_interval (int): 更新间隔，单位秒
        max_iterations (int): 最大循环次数（实际应用中可设为无限循环）
        use_mock (bool): 是否使用模拟数据（用于演示）
        """
        print(f"\n开始监控 {len(self.symbols)} 只股票...")
        print(f"更新间隔: {update_interval} 秒")
        print(f"买入信号阈值: {self.threshold}\n")
        
        # 初始检查
        self.check_signals()
        
        # 模拟循环更新
        for i in range(max_iterations - 1):
            print(f"\n等待 {update_interval} 秒后进行下一次更新... (按Ctrl+C终止)")
            
            try:
                # 在实际应用中，这里会等待到下一个更新时间点
                # 为了示例方便，这里只简单等待几秒钟
                time.sleep(3)  # 模拟等待，实际应用中使用update_interval
                
                # 更新数据
                self.update_data(use_mock=use_mock)
                
                # 检查信号
                self.check_signals()
                
            except KeyboardInterrupt:
                print("\n监控被用户中断")
                break
            
        print("\n模拟监控完成")

def main():
    """运行实时评分示例"""
    print("多因子买入信号评分系统 - 实时监控示例")
    print("-" * 50)
    
    # 设置要监控的股票列表
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # 创建监控器
    monitor = RealTimeMonitor(symbols, threshold=60)
    
    # 初始化数据
    try:
        monitor.initialize(days_lookback=365)
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        print("生成样本数据用于演示...")
        
        # 如果实际数据加载失败，创建模拟数据
        for symbol in symbols:
            if symbol not in monitor.data:
                data = generate_sample_data(days=252)
                monitor.data[symbol] = data
                monitor.scorers[symbol] = BuySignalScorer(data)
                monitor.latest_scores[symbol] = monitor.scorers[symbol].calculate_buy_signal_score()
                print(f"为 {symbol} 生成了模拟数据")
    
    # 运行监控
    monitor.run_monitor(update_interval=3600, max_iterations=3, use_mock=True)

if __name__ == "__main__":
    main()