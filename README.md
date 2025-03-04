# 多因子买入信号评分系统

这个项目提供了一个基于多因子技术分析的股票买入信号评分系统，通过整合多种技术指标来生成综合性的买入信号强度评分。

## 功能特点

- **多因子综合评分**：结合多种技术指标，生成0-100分的综合评分
- **灵活的数据加载**：支持从Yahoo Finance、CSV文件和自定义API加载数据
- **完整的技术指标计算**：内置丰富的技术指标计算函数
- **可视化分析**：提供多种可视化工具，直观展示买入信号
- **数据预处理**：内置数据清洗和预处理功能
- **支持回测**：可用于历史数据回测

## 安装指南

```bash
pip install trading-signal-scorer
```

或从源码安装：

```bash
git clone https://github.com/yourusername/trading-signal-scorer.git
cd trading-signal-scorer
pip install -e .
```

### 运行条件

- Python >= 3.7
- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- yfinance
- ta-lib (需要先安装TA-Lib C库)

## 快速入门

```python
from trading_signal_scorer.data_loader import load_from_yahoo
from trading_signal_scorer.buy_signal_scorer import BuySignalScorer
from trading_signal_scorer.plot_utils import plot_price_with_signals

# 加载股票数据
data = load_from_yahoo("AAPL", period="1y")

# 创建评分器并计算买入信号
scorer = BuySignalScorer(data)
signal = scorer.calculate_buy_signal_score()

# 输出评分结果
print(f"买入信号强度: {signal['total_score']:.2f}/100")
print(f"信号解读: {signal['signal_strength']}")

# 绘制带有买入信号的价格图表
fig = plot_price_with_signals(data, scorer, threshold=50)
fig.savefig("buy_signals.png")
```

## 评分系统说明

买入信号评分由六个主要组成部分构成，总分为100分：

1. **RSI指标**（0-20分）：基于RSI超卖情况和底背离进行评分
2. **价格形态**（0-20分）：基于布林带、移动平均线和蜡烛图形态评分
3. **成交量分析**（0-15分）：基于相对成交量和OBV指标评分
4. **支撑位分析**（0-15分）：基于技术支撑位和斐波那契回调位评分
5. **动量指标**（0-15分）：基于随机指标、威廉指标和MACD评分
6. **波动率分析**（0-15分）：基于ATR和布林带宽度评分

评分标准：
- 80-100分：极强买入信号
- 70-79分：强买入信号
- 60-69分：中强买入信号
- 50-59分：中等买入信号
- 40-49分：弱买入信号
- 30-39分：极弱买入信号
- 0-29分：无买入信号

## 使用示例

详细的使用示例请参考 `examples/` 目录下的示例文件。

## 许可证

MIT