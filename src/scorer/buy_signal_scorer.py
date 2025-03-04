"""
买入信号评分器 - 基于多个技术指标评估股票买入时机
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import traceback

# 导入指标计算函数
try:
    from ..indicators import (
        calculate_rsi, calculate_bollinger_bands, calculate_moving_averages,
        calculate_volume_indicators, calculate_volatility_indicators, 
        calculate_momentum_indicators, identify_support_levels,
        identify_candlestick_patterns, detect_divergences
    )
except ImportError:
    # 处理相对导入失败的情况（例如直接运行此文件）
    try:
        from src.indicators import (
            calculate_rsi, calculate_bollinger_bands, calculate_moving_averages,
            calculate_volume_indicators, calculate_volatility_indicators, 
            calculate_momentum_indicators, identify_support_levels,
            identify_candlestick_patterns, detect_divergences
        )
    except ImportError:
        # 最后尝试从当前目录导入
        from indicators import (
            calculate_rsi, calculate_bollinger_bands, calculate_moving_averages,
            calculate_volume_indicators, calculate_volatility_indicators, 
            calculate_momentum_indicators, identify_support_levels,
            identify_candlestick_patterns, detect_divergences
        )

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BuySignalScorer:
    """
    多因子买入信号评分器，基于六个维度的技术指标综合评分
    """
    
    def __init__(self, data, use_talib=True, custom_weights=None):
        """
        初始化评分器
        
        参数:
        data (pandas.DataFrame): 价格数据，包含date, open, high, low, close, volume列
        use_talib (bool): 是否使用TA-Lib库计算指标（如果为False，使用内部实现）
        custom_weights (dict): 自定义各组成部分的权重
        """
        self.data = data.copy()
        self.use_talib = use_talib
        
        # 默认权重
        self.weights = {
            'RSI指标': 20,  # 0-20分
            '价格形态': 20,  # 0-20分
            '成交量分析': 15,  # 0-15分
            '支撑位分析': 15,  # 0-15分
            '动量指标': 15,  # 0-15分
            '波动率分析': 15,  # 0-15分
        }
        
        # 如果提供了自定义权重，更新默认权重
        if custom_weights:
            for component, weight in custom_weights.items():
                if component in self.weights:
                    self.weights[component] = weight
            
            # 确保总权重为100
            total_weight = sum(self.weights.values())
            if total_weight != 100:
                # 按比例调整所有权重
                factor = 100 / total_weight
                for component in self.weights:
                    self.weights[component] = round(self.weights[component] * factor)
        
        try:
            # 计算所有技术指标
            self._calculate_indicators()
            logger.info("初始化买入信号评分器完成")
        except Exception as e:
            logger.error(f"初始化评分器时出错: {str(e)}")
            traceback.print_exc()
            raise
    
    def _calculate_indicators(self):
        """计算评分所需的所有技术指标"""
        logger.info("开始计算技术指标")
        
        try:
            # RSI指标
            self.data = calculate_rsi(self.data, periods=[6, 14, 28])
            
            # 布林带
            self.data = calculate_bollinger_bands(self.data, period=20, std_devs=[2, 3])
            
            # 移动平均线
            self.data = calculate_moving_averages(self.data, periods=[10, 20, 50, 200], types=['sma', 'ema'])
            
            # 成交量指标
            self.data = calculate_volume_indicators(self.data, periods=[10, 20])
            
            # 波动率指标
            self.data = calculate_volatility_indicators(self.data)
            
            # 动量指标
            self.data = calculate_momentum_indicators(self.data)
            
            # 蜡烛图形态
            self.data = identify_candlestick_patterns(self.data)
            
            # 检测背离
            self.data = detect_divergences(self.data)
            
            # 计算支撑位和斐波那契回调位
            self.support_levels, self.fib_levels = identify_support_levels(self.data)
            
            logger.info("技术指标计算完成")
        
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            traceback.print_exc()
            raise
    
    def _score_rsi(self, index):
        """
        计算RSI指标分量得分 (0-20分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: RSI指标得分 (0-20)
        """
        # 获取当前RSI值
        rsi6 = self.data['rsi_6'].iloc[index]
        rsi14 = self.data['rsi_14'].iloc[index]
        rsi28 = self.data['rsi_28'].iloc[index]
        
        # 初始得分
        score = 0
        
        # 1. RSI超卖评分 (0-10分)
        # RSI值越低，得分越高
        if rsi14 <= 30:
            # 在30-20区间，分数线性增加
            if rsi14 > 20:
                score += 5 + (30 - rsi14) / 10 * 5
            # 低于20分，满分
            else:
                score += 10
        # 30-40区间，给予低分
        elif rsi14 <= 40:
            score += (40 - rsi14) / 10 * 5
        
        # 2. 多周期RSI一致性 (0-5分)
        # 检查不同周期的RSI是否都处于低位
        rsi_avg = (rsi6 + rsi14 + rsi28) / 3
        if rsi_avg < 40:
            # 计算RSI之间的差异
            rsi_diff = max(rsi6, rsi14, rsi28) - min(rsi6, rsi14, rsi28)
            # 差异越小，一致性越高
            if rsi_diff < 5:
                score += 5
            elif rsi_diff < 10:
                score += 3
            elif rsi_diff < 15:
                score += 1
        
        # 3. RSI底背离 (0-5分)
        if self.data['rsi_bullish_divergence'].iloc[index]:
            score += 5
        
        # 确保分数在0-20范围内
        return min(20, max(0, score))
    
    def _score_price_pattern(self, index):
        """
        计算价格形态分量得分 (0-20分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: 价格形态得分 (0-20)
        """
        score = 0
        
        # 获取当前价格和移动平均线
        current_price = self.data['close'].iloc[index]
        
        # 1. 布林带指标 (0-8分)
        # 价格触及或低于下轨得分高
        bb_lower = self.data['bb_lower'].iloc[index]
        bb_lower_3std = self.data['bb_lower_3std'].iloc[index]
        bb_width = self.data['bb_width'].iloc[index]
        
        # 与布林带的关系
        if current_price <= bb_lower_3std:
            # 触及3倍标准差下轨，极度超卖
            score += 8
        elif current_price <= bb_lower:
            # 触及2倍标准差下轨
            score += 6
            # 如果布林带收缩后触及下轨，额外加分
            if bb_width < self.data['bb_width'].rolling(window=20).mean().iloc[index]:
                score += 1
        elif current_price <= bb_lower * 1.01:
            # 接近下轨
            score += 4
        
        # 2. 移动平均线支撑 (0-7分)
        # 检查价格是否接近或反弹自重要均线
        sma_50 = self.data['sma_50'].iloc[index] if 'sma_50' in self.data.columns else None
        sma_200 = self.data['sma_200'].iloc[index] if 'sma_200' in self.data.columns else None
        
        if sma_50 is not None:
            # 价格在50日均线附近
            price_to_sma50_ratio = current_price / sma_50
            if 0.98 <= price_to_sma50_ratio <= 1.02:
                score += 3
            elif 0.95 <= price_to_sma50_ratio < 0.98:
                score += 2
        
        if sma_200 is not None:
            # 价格在200日均线附近
            price_to_sma200_ratio = current_price / sma_200
            if 0.98 <= price_to_sma200_ratio <= 1.02:
                score += 4
            elif 0.95 <= price_to_sma200_ratio < 0.98:
                score += 3
        
        # 3. 蜡烛图形态 (0-5分)
        # 检查看涨蜡烛图形态
        if self.data['hammer'].iloc[index]:
            # 锤子线
            score += 4
        elif self.data['long_lower_shadow'].iloc[index]:
            # 长下影线
            score += 3
        elif self.data['doji'].iloc[index] and index > 0 and self.data['close'].iloc[index-1] < self.data['open'].iloc[index-1]:
            # 十字星形态在下跌后出现
            score += 2
        
        # 确保分数在0-20范围内
        return min(20, max(0, score))
    
    def _score_volume(self, index):
        """
        计算成交量分量得分 (0-15分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: 成交量得分 (0-15)
        """
        score = 0
        
        # 1. 相对成交量评估 (0-5分)
        # 当前成交量相对于20日均量
        if 'relative_volume' in self.data.columns:
            rel_volume = self.data['relative_volume'].iloc[index]
            
            # 成交量放大
            if rel_volume > 2.0:
                score += 5
            elif rel_volume > 1.5:
                score += 4
            elif rel_volume > 1.2:
                score += 3
            elif rel_volume > 1.0:
                score += 2
        
        # 2. OBV底背离 (0-7分)
        # OBV指标与价格的底背离
        if 'obv_bullish_divergence' in self.data.columns and self.data['obv_bullish_divergence'].iloc[index]:
            score += 7
        
        # 3. 成交量趋势与价格关系 (0-3分)
        # 检查成交量是否在价格下跌时萎缩
        if index >= 5:
            # 计算最近5天的价格变化
            price_change = (self.data['close'].iloc[index] / self.data['close'].iloc[index-5] - 1) * 100
            
            # 计算成交量变化
            volume_change = (self.data['volume'].iloc[index] / self.data['volume'].iloc[index-5] - 1) * 100
            
            # 价格下跌但成交量萎缩（可能接近底部）
            if price_change < -3 and volume_change < -10:
                score += 3
            elif price_change < -2 and volume_change < -5:
                score += 2
        
        # 确保分数在0-15范围内
        return min(15, max(0, score))
    
    def _score_support_resistance(self, index):
        """
        计算支撑位分量得分 (0-15分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: 支撑位得分 (0-15)
        """
        score = 0
        current_price = self.data['close'].iloc[index]
        
        # 1. 历史支撑位 (0-8分)
        # 检查当前价格是否接近支撑位
        if self.support_levels:
            # 计算价格与最近支撑位的距离
            closest_support = min(self.support_levels, key=lambda x: abs(x - current_price))
            support_distance = abs(current_price - closest_support) / current_price * 100
            
            # 价格非常接近支撑位
            if support_distance < 1:
                score += 8
            elif support_distance < 3:
                score += 6
            elif support_distance < 5:
                score += 4
            elif support_distance < 7:
                score += 2
        
        # 2. 斐波那契回调位 (0-7分)
        # 检查价格是否接近重要的斐波那契回调位
        if self.fib_levels:
            # 计算价格与斐波那契回调位的距离
            fib_distances = {}
            for level_name, level_price in self.fib_levels.items():
                # 计算百分比距离
                distance = abs(current_price - level_price) / current_price * 100
                fib_distances[level_name] = distance
            
            # 获取最接近的斐波那契水平
            closest_fib = min(fib_distances, key=fib_distances.get)
            closest_distance = fib_distances[closest_fib]
            
            # 根据接近程度和斐波那契级别给分
            if closest_distance < 2:
                # 非常接近某个斐波那契水平
                if closest_fib in ['fib_0.618', 'fib_0.786']:
                    # 黄金比例和重要回调位
                    score += 7
                elif closest_fib in ['fib_0.5', 'fib_0.382']:
                    # 其他主要回调位
                    score += 5
                else:
                    # 次要回调位
                    score += 3
            elif closest_distance < 5:
                # 稍微接近某个斐波那契水平
                if closest_fib in ['fib_0.618', 'fib_0.786']:
                    score += 4
                elif closest_fib in ['fib_0.5', 'fib_0.382']:
                    score += 3
                else:
                    score += 1
        
        # 确保分数在0-15范围内
        return min(15, max(0, score))
    
    def _score_momentum(self, index):
        """
        计算动量指标分量得分 (0-15分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: 动量指标得分 (0-15)
        """
        score = 0
        
        # 1. 随机指标评估 (0-5分)
        if 'slowk' in self.data.columns and 'slowd' in self.data.columns:
            slowk = self.data['slowk'].iloc[index]
            slowd = self.data['slowd'].iloc[index]
            
            # 超卖区域
            if slowk < 20 and slowd < 20:
                score += 4
                # 金叉形成
                if index > 0 and slowk > slowd and self.data['slowk'].iloc[index-1] <= self.data['slowd'].iloc[index-1]:
                    score += 1
            elif slowk < 30 and slowd < 30:
                score += 2
        
        # 2. 威廉指标评估 (0-4分)
        if 'willr' in self.data.columns:
            willr = self.data['willr'].iloc[index]
            
            # 超卖区域（威廉指标为负，越接近-100越超卖）
            if willr < -80:
                score += 4
            elif willr < -70:
                score += 3
            elif willr < -60:
                score += 1
        
        # 3. MACD评估 (0-6分)
        if 'macd' in self.data.columns and 'macd_signal' in self.data.columns and 'macd_hist' in self.data.columns:
            macd = self.data['macd'].iloc[index]
            macd_signal = self.data['macd_signal'].iloc[index]
            macd_hist = self.data['macd_hist'].iloc[index]
            
            # MACD底背离
            if 'macd_bullish_divergence' in self.data.columns and self.data['macd_bullish_divergence'].iloc[index]:
                score += 6
            # MACD金叉
            elif index > 0 and macd > macd_signal and self.data['macd'].iloc[index-1] <= self.data['macd_signal'].iloc[index-1]:
                score += 5
            # MACD柱状图由负转正
            elif index > 0 and macd_hist > 0 and self.data['macd_hist'].iloc[index-1] <= 0:
                score += 4
            # MACD接近零轴
            elif abs(macd) < 0.01 * self.data['close'].iloc[index]:
                score += 2
        
        # 确保分数在0-15范围内
        return min(15, max(0, score))
    
    def _score_volatility(self, index):
        """
        计算波动率分量得分 (0-15分)
        
        参数:
        index (int): 数据点索引
        
        返回:
        float: 波动率得分 (0-15)
        """
        score = 0
        
        # 1. ATR相对值评估 (0-6分)
        if 'relative_atr' in self.data.columns:
            rel_atr = self.data['relative_atr'].iloc[index]
            
            # 计算30日平均相对ATR
            if index >= 30:
                avg_rel_atr = self.data['relative_atr'].iloc[index-29:index+1].mean()
                
                # 波动率显著低于平均水平（可能即将反转）
                if rel_atr < avg_rel_atr * 0.6:
                    score += 6
                elif rel_atr < avg_rel_atr * 0.7:
                    score += 5
                elif rel_atr < avg_rel_atr * 0.8:
                    score += 4
                elif rel_atr < avg_rel_atr * 0.9:
                    score += 2
        
        # 2. 布林带宽度变化 (0-5分)
        if 'bb_width' in self.data.columns:
            bb_width = self.data['bb_width'].iloc[index]
            
            # 计算20日平均布林带宽度
            if index >= 20:
                avg_bb_width = self.data['bb_width'].iloc[index-19:index+1].mean()
                
                # 布林带收缩（可能即将爆发）
                if bb_width < avg_bb_width * 0.6:
                    score += 5
                elif bb_width < avg_bb_width * 0.7:
                    score += 4
                elif bb_width < avg_bb_width * 0.8:
                    score += 3
                elif bb_width < avg_bb_width * 0.9:
                    score += 1
        
        # 3. 波动率趋势 (0-4分)
        if 'atr_change' in self.data.columns:
            atr_change = self.data['atr_change'].iloc[index]
            
            # 波动率从高位回落（可能已完成抛售）
            if atr_change < -30:
                score += 4
            elif atr_change < -20:
                score += 3
            elif atr_change < -10:
                score += 2
            elif atr_change < -5:
                score += 1
        
        # 确保分数在0-15范围内
        return min(15, max(0, score))
    
    def calculate_buy_signal_score(self, index=None):
        """
        计算买入信号综合评分
        
        参数:
        index (int): 可选，指定数据点索引，默认为最后一个点
        
        返回:
        dict: 包含总分和各组成部分分数的字典
        """
        try:
            # 如果未指定索引，使用最后一个数据点
            if index is None:
                index = len(self.data) - 1
            
            # 确保索引有效
            if index < 0:
                index = len(self.data) + index
            
            if index < 0 or index >= len(self.data):
                raise ValueError(f"索引 {index} 超出数据范围 (0-{len(self.data)-1})")
            
            # 计算各组成部分的评分
            component_scores = {
                'RSI指标': self._score_rsi(index),
                '价格形态': self._score_price_pattern(index),
                '成交量分析': self._score_volume(index),
                '支撑位分析': self._score_support_resistance(index),
                '动量指标': self._score_momentum(index),
                '波动率分析': self._score_volatility(index)
            }
            
            # 计算总分
            total_score = sum(component_scores.values())
            
            # 确保总分在0-100范围内
            total_score = min(100, max(0, total_score))
            
            # 确定信号强度描述
            signal_strength = self._get_signal_strength(total_score)
            
            # 创建结果字典
            result = {
                'date': self.data['date'].iloc[index],
                'price': self.data['close'].iloc[index],
                'total_score': total_score,
                'component_scores': component_scores,
                'signal_strength': signal_strength
            }
            
            logger.info(f"计算完成，日期: {result['date']}, 总分: {total_score:.2f}, 信号强度: {signal_strength}")
            
            return result
        
        except Exception as e:
            logger.error(f"计算买入信号评分时出错: {str(e)}")
            traceback.print_exc()
            raise
    
    def _get_signal_strength(self, score):
        """
        根据总分确定信号强度描述
        
        参数:
        score (float): 总分 (0-100)
        
        返回:
        str: 信号强度描述
        """
        if score >= 80:
            return "极强买入信号"
        elif score >= 70:
            return "强买入信号"
        elif score >= 60:
            return "中强买入信号"
        elif score >= 50:
            return "中等买入信号"
        elif score >= 40:
            return "弱买入信号"
        elif score >= 30:
            return "极弱买入信号"
        else:
            return "无买入信号"
    
    def evaluate_recent_signals(self, days=30):
        """
        评估最近一段时间的信号强度
        
        参数:
        days (int): 评估的天数
        
        返回:
        pandas.DataFrame: 包含日期、价格和信号得分的DataFrame
        """
        try:
            logger.info(f"评估最近 {days} 天的买入信号")
            
            # 确保天数有效
            days = min(days, len(self.data))
            
            # 创建结果列表
            signals = []
            
            # 计算最近days天的信号
            for i in range(len(self.data) - days, len(self.data)):
                score_data = self.calculate_buy_signal_score(i)
                signals.append({
                    'date': score_data['date'],
                    'price': score_data['price'],
                    'total_score': score_data['total_score'],
                    'signal_strength': score_data['signal_strength']
                })
            
            # 转换为DataFrame
            signals_df = pd.DataFrame(signals)
            
            logger.info(f"评估完成，共 {len(signals_df)} 条评分记录")
            return signals_df
        
        except Exception as e:
            logger.error(f"评估最近信号时出错: {str(e)}")
            traceback.print_exc()
            raise
    
    def get_historical_scores(self, start_index=0, end_index=None):
        """
        计算指定时间段内的历史信号评分
        
        参数:
        start_index (int): 起始索引
        end_index (int): 结束索引，默认为最后一个点
        
        返回:
        pandas.DataFrame: 包含日期、价格和信号得分的DataFrame
        """
        try:
            logger.info("计算历史信号评分")
            
            # 设置默认结束索引
            if end_index is None:
                end_index = len(self.data) - 1
            
            # 确保索引有效
            start_index = max(0, start_index)
            end_index = min(len(self.data) - 1, end_index)
            
            # 计算范围内的每日评分
            scores = []
            for i in range(start_index, end_index + 1):
                score_data = self.calculate_buy_signal_score(i)
                scores.append({
                    'date': score_data['date'],
                    'price': score_data['price'],
                    'total_score': score_data['total_score'],
                    'signal_strength': score_data['signal_strength'],
                    'rsi_score': score_data['component_scores']['RSI指标'],
                    'price_pattern_score': score_data['component_scores']['价格形态'],
                    'volume_score': score_data['component_scores']['成交量分析'],
                    'support_score': score_data['component_scores']['支撑位分析'],
                    'momentum_score': score_data['component_scores']['动量指标'],
                    'volatility_score': score_data['component_scores']['波动率分析']
                })
            
            # 转换为DataFrame
            scores_df = pd.DataFrame(scores)
            
            logger.info(f"历史评分计算完成，共 {len(scores_df)} 条记录")
            return scores_df
        
        except Exception as e:
            logger.error(f"计算历史评分时出错: {str(e)}")
            traceback.print_exc()
            raise
    
    def find_best_signals(self, threshold=70, max_results=5):
        """
        查找最佳买入信号点
        
        参数:
        threshold (float): 信号阈值
        max_results (int): 最大结果数量
        
        返回:
        list: 包含最佳信号的字典列表
        """
        try:
            logger.info(f"查找信号评分大于 {threshold} 的买入点")
            
            # 计算所有点的评分
            all_scores = []
            for i in range(len(self.data)):
                score_data = self.calculate_buy_signal_score(i)
                if score_data['total_score'] >= threshold:
                    all_scores.append(score_data)
            
            # 按评分降序排序
            all_scores.sort(key=lambda x: x['total_score'], reverse=True)
            
            # 返回前N个结果
            top_signals = all_scores[:max_results]
            
            logger.info(f"找到 {len(top_signals)} 个满足条件的买入信号")
            return top_signals
        
        except Exception as e:
            logger.error(f"查找最佳买入信号时出错: {str(e)}")
            traceback.print_exc()
            raise