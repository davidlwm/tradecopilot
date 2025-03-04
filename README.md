# T型交易低点识别系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于多因子分析的交易低点识别系统，专为T型交易(高抛低吸)策略设计，特别适用于波动性较大的股票。

## 功能特点

- 六维度综合分析：RSI指标、价格形态、成交量、支撑位、动量指标、波动率
- 0-100分买入信号强度评分
- 多级别信号强度分类
- 详细的分项得分明细
- 历史信号强度评估
- 支持自定义参数调整

## 安装

```bash
pip install -r requirements.txt
python setup.py install