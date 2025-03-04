# 安装指南

本文档提供多因子买入信号评分系统的详细安装说明。

## 系统要求

- Python 3.7+
- 64位操作系统 (Windows, macOS 或 Linux)
- 足够的磁盘空间 (至少200MB)
- 互联网连接 (用于从Yahoo Finance下载数据)

## 依赖项

主要依赖项：

- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- yfinance
- ta-lib (需要特殊安装)
- requests

## 基本安装

### 使用pip安装

最简单的安装方式是使用pip直接安装：

```bash
pip install trading-signal-scorer
```

### 从源码安装

也可以从源码安装：

```bash
git clone https://github.com/yourusername/trading-signal-scorer.git
cd trading-signal-scorer
pip install -e .
```

## TA-Lib的特殊安装说明

TA-Lib是一个技术分析库，需要先安装C/C++库，然后再安装Python包装器。

### Windows

1. 从 [这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) 下载预编译的wheel文件
2. 安装下载的wheel文件：
   ```
   pip install TA_Lib‑0.4.0‑cp39‑cp39‑win_amd64.whl
   ```
   (选择与你的Python版本匹配的wheel文件)

### macOS

使用Homebrew安装：

```bash
brew install ta-lib
pip install ta-lib
```

### Linux (Ubuntu/Debian)

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install build-essential

# 下载并安装TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.