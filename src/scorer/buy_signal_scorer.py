import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime, timedelta

class BuySignalScorer:
    """
    基于多因子的买入信号强度评分系统
    满分100分，最低0分
    """
    
    def __init__(self, price_data):
        """
        初始化评分系统
        
        参数:
        price_data (pandas DataFrame): 包含 'date', 'open', 'high', 'low', 'close', 'volume' 的DataFrame
        """
        self.data = price_data
        self.prepare_data()
        
    def prepare_data(self):
        """准备计算所需的技术指标"""
        # 确保数据是按日期排序的
        self.data = self.data.sort_values('date')
        
        # 计算基础技术指标
        self._calc_price_indicators()
        self._calc_volume_indicators()
        self._calc_volatility_indicators()
        self._calc_momentum_indicators()
        self._calc_support_resistance()
        
    def _calc_price_indicators(self):
        """计算价格相关指标"""
        # 布林带 (20,2)
        self.data['upper_band'], self.data['middle_band'], self.data['lower_band'] = ta.BBANDS(
            self.data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # 布林带 (20,3) - 用于极端情况
        self.data['upper_band_extreme'], _, self.data['lower_band_extreme'] = ta.BBANDS(
            self.data['close'], timeperiod=20, nbdevup=3, nbdevdn=3
        )
        
        # 移动平均线
        self.data['sma_10'] = ta.SMA(self.data['close'], timeperiod=10)
        self.data['sma_20'] = ta.SMA(self.data['close'], timeperiod=20)
        self.data['sma_50'] = ta.SMA(self.data['close'], timeperiod=50)
        self.data['sma_200'] = ta.SMA(self.data['close'], timeperiod=200)
        
        # EMA
        self.data['ema_10'] = ta.EMA(self.data['close'], timeperiod=10)
        self.data['ema_20'] = ta.EMA(self.data['close'], timeperiod=20)
        self.data['ema_50'] = ta.EMA(self.data['close'], timeperiod=50)
        
        # MACD
        self.data['macd'], self.data['macd_signal'], self.data['macd_hist'] = ta.MACD(
            self.data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
    def _calc_volume_indicators(self):
        """计算成交量相关指标"""
        # 成交量移动平均
        self.data['volume_sma_10'] = ta.SMA(self.data['volume'], timeperiod=10)
        self.data['volume_sma_20'] = ta.SMA(self.data['volume'], timeperiod=20)
        
        # 计算相对成交量 (当日成交量/20日平均成交量)
        self.data['relative_volume'] = self.data['volume'] / self.data['volume_sma_20']
        
        # 成交量波动率 (20日成交量标准差/20日平均成交量)
        volumes = self.data['volume'].rolling(window=20).std()
        self.data['volume_volatility'] = volumes / self.data['volume_sma_20']
        
        # OBV (累积能量潮)
        self.data['obv'] = ta.OBV(self.data['close'], self.data['volume'])
        
    def _calc_volatility_indicators(self):
        """计算波动率相关指标"""
        # ATR - 平均真实波幅
        self.data['atr'] = ta.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)
        
        # 相对ATR (ATR/收盘价)
        self.data['relative_atr'] = self.data['atr'] / self.data['close'] * 100
        
        # ATR变化率
        self.data['atr_change'] = self.data['atr'].pct_change(10) * 100
        
        # 布林带宽度 (作为波动率指标)
        self.data['bb_width'] = (self.data['upper_band'] - self.data['lower_band']) / self.data['middle_band'] * 100
        
    def _calc_momentum_indicators(self):
        """计算动量相关指标"""
        # RSI - 多周期
        self.data['rsi_6'] = ta.RSI(self.data['close'], timeperiod=6)
        self.data['rsi_14'] = ta.RSI(self.data['close'], timeperiod=14)
        self.data['rsi_28'] = ta.RSI(self.data['close'], timeperiod=28)
        
        # 随机指标
        self.data['slowk'], self.data['slowd'] = ta.STOCH(
            self.data['high'], self.data['low'], self.data['close'],
            fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        
        # Williams %R
        self.data['willr'] = ta.WILLR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)
        
        # CCI - 商品通道指数
        self.data['cci'] = ta.CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=20)
        
    def _calc_support_resistance(self):
        """识别支撑位和阻力位"""
        # 仅使用最近100个数据点来计算支撑阻力位
        recent_data = self.data.tail(100)
        
        # 定义接近度百分比
        proximity_pct = 0.02  # 2%
        
        # 寻找局部极小值作为支撑位
        self.support_levels = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                self.support_levels.append(recent_data['low'].iloc[i])
        
        # 计算斐波那契回调位
        if len(recent_data) > 0:
            max_price = recent_data['high'].max()
            min_price = recent_data['low'].min()
            price_range = max_price - min_price
            
            self.fib_levels = {
                'fib_0': min_price,
                'fib_0.236': min_price + 0.236 * price_range,
                'fib_0.382': min_price + 0.382 * price_range,
                'fib_0.5': min_price + 0.5 * price_range,
                'fib_0.618': min_price + 0.618 * price_range,
                'fib_0.786': min_price + 0.786 * price_range,
                'fib_1': max_price
            }
    
    def _score_rsi(self, current_idx):
        """RSI打分组件 (0-20分)"""
        score = 0
        
        # 当前各周期RSI值
        rsi_6 = self.data['rsi_6'].iloc[current_idx]
        rsi_14 = self.data['rsi_14'].iloc[current_idx]
        rsi_28 = self.data['rsi_28'].iloc[current_idx]
        
        # 基于RSI超卖区评分
        if rsi_14 < 30:
            if rsi_14 < 20:
                score += 10  # 严重超卖
            else:
                score += 7   # 一般超卖
        elif rsi_14 < 40:
            score += 3       # 接近超卖
            
        # 多周期RSI一致性
        if rsi_6 < 30 and rsi_14 < 35 and rsi_28 < 40:
            score += 5       # 三个周期都显示超卖
        elif (rsi_6 < 30 and rsi_14 < 35) or (rsi_14 < 35 and rsi_28 < 40):
            score += 3       # 两个周期显示超卖
            
        # RSI底部背离
        if current_idx > 10:
            # 如果价格创新低，但RSI没有
            if (self.data['close'].iloc[current_idx] < self.data['close'].iloc[current_idx-10:current_idx].min() and
                rsi_14 > self.data['rsi_14'].iloc[current_idx-10:current_idx].min()):
                score += 5   # RSI底背离
                
        return min(score, 20)  # 最高20分
        
    def _score_price_pattern(self, current_idx):
        """价格形态打分组件 (0-20分)"""
        score = 0
        
        current_price = self.data['close'].iloc[current_idx]
        
        # 布林带相关信号
        if current_price <= self.data['lower_band'].iloc[current_idx]:
            score += 5       # 触及下轨
            
            if current_price <= self.data['lower_band_extreme'].iloc[current_idx]:
                score += 3   # 触及极端下轨 (3倍标准差)
                
        # 布林带收缩后突破
        if current_idx > 10:
            bb_width_prev = self.data['bb_width'].iloc[current_idx-10:current_idx].mean()
            bb_width_current = self.data['bb_width'].iloc[current_idx]
            
            if bb_width_current < bb_width_prev * 0.7 and current_price <= self.data['lower_band'].iloc[current_idx]:
                score += 3   # 布林带收缩后触及下轨
        
        # 移动平均线支撑
        if (current_price <= self.data['sma_50'].iloc[current_idx] * 1.01 and 
            current_price >= self.data['sma_50'].iloc[current_idx] * 0.99):
            score += 3       # 接近50日均线支撑
            
        if (current_price <= self.data['sma_200'].iloc[current_idx] * 1.01 and
            current_price >= self.data['sma_200'].iloc[current_idx] * 0.99):
            score += 4       # 接近200日均线支撑
            
        # 蜡烛图形态 (简化版)
        if current_idx >= 1:
            # 长下影线 (潜在反转信号)
            body_size = abs(self.data['open'].iloc[current_idx] - self.data['close'].iloc[current_idx])
            lower_shadow = min(self.data['open'].iloc[current_idx], self.data['close'].iloc[current_idx]) - self.data['low'].iloc[current_idx]
            
            if lower_shadow > body_size * 2:
                score += 4   # 长下影线
                
            # 锤子线形态
            if (lower_shadow > body_size * 1.5 and
                body_size < (self.data['high'].iloc[current_idx] - self.data['low'].iloc[current_idx]) * 0.3):
                score += 2   # 锤子线特征
                
        return min(score, 20)  # 最高20分
        
    def _score_volume(self, current_idx):
        """成交量打分组件 (0-15分)"""
        score = 0
        
        current_volume = self.data['volume'].iloc[current_idx]
        relative_vol = self.data['relative_volume'].iloc[current_idx]
        
        # 放量情况评分
        if relative_vol > 2.0:
            score += 8       # 成交量是20日均量的2倍以上
        elif relative_vol > 1.5:
            score += 6       # 成交量是20日均量的1.5-2倍
        elif relative_vol > 1.2:
            score += 3       # 成交量是20日均量的1.2-1.5倍
            
        # OBV底背离
        if current_idx > 10:
            # 如果价格创新低，但OBV没有
            if (self.data['close'].iloc[current_idx] < self.data['close'].iloc[current_idx-10:current_idx].min() and
                self.data['obv'].iloc[current_idx] > self.data['obv'].iloc[current_idx-10:current_idx].min()):
                score += 7   # OBV底背离
                
        return min(score, 15)  # 最高15分
        
    def _score_support_resistance(self, current_idx):
        """支撑位评分组件 (0-15分)"""
        score = 0
        
        current_price = self.data['close'].iloc[current_idx]
        proximity_threshold = current_price * 0.02  # 2%接近度阈值
        
        # 检查是否接近支撑位
        for support in self.support_levels:
            if abs(current_price - support) < proximity_threshold:
                score += 7
                break
                
        # 检查是否接近斐波那契回调位
        near_fib = False
        for level_name in ['fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786']:
            fib_level = self.fib_levels[level_name]
            if abs(current_price - fib_level) < proximity_threshold:
                if level_name in ['fib_0.618', 'fib_0.786']:  # 黄金分割和0.786是更重要的回调位
                    score += 8
                else:
                    score += 5
                near_fib = True
                break
                
        # 如果同时接近支撑位和斐波那契回调位，额外加分
        if score > 7 and near_fib:
            score += 3  # 支撑叠加
            
        return min(score, 15)  # 最高15分
        
    def _score_momentum(self, current_idx):
        """动量指标打分组件 (0-15分)"""
        score = 0
        
        # 随机指标评分
        if self.data['slowk'].iloc[current_idx] < 20 and self.data['slowd'].iloc[current_idx] < 20:
            score += 4  # K和D都在超卖区
            if self.data['slowk'].iloc[current_idx] > self.data['slowd'].iloc[current_idx]:
                score += 3  # 金叉形成
                
        # Williams %R评分
        willr = self.data['willr'].iloc[current_idx]
        if willr < -80:
            score += 3
            if willr < -90:
                score += 2  # 极度超卖
                
        # CCI评分
        cci = self.data['cci'].iloc[current_idx]
        if cci < -100:
            score += 2
            if cci < -200:
                score += 3  # 极度超卖
                
        # MACD底背离
        if current_idx > 20:
            # 如果价格创新低，但MACD直方图没有
            price_new_low = self.data['close'].iloc[current_idx] < self.data['close'].iloc[current_idx-20:current_idx].min()
            macd_hist_not_new_low = self.data['macd_hist'].iloc[current_idx] > self.data['macd_hist'].iloc[current_idx-20:current_idx].min()
            
            if price_new_low and macd_hist_not_new_low:
                score += 5  # MACD底背离
                
        # MACD零轴附近
        if abs(self.data['macd'].iloc[current_idx]) < 0.01 * self.data['close'].iloc[current_idx]:
            score += 2  # MACD接近零轴
            
        return min(score, 15)  # 最高15分
        
    def _score_volatility(self, current_idx):
        """波动率打分组件 (0-15分)"""
        score = 0
        
        # 相对ATR评分 (波动收缩可能预示反转)
        if current_idx > 30:
            atr_30d_avg = self.data['relative_atr'].iloc[current_idx-30:current_idx].mean()
            current_rel_atr = self.data['relative_atr'].iloc[current_idx]
            
            if current_rel_atr < atr_30d_avg * 0.7:
                score += 5  # 波动率显著收缩
            elif current_rel_atr < atr_30d_avg * 0.85:
                score += 3  # 波动率适度收缩
                
        # 布林带宽度收缩 (可能是蓄势待发)
        if current_idx > 20:
            bb_width_20d_avg = self.data['bb_width'].iloc[current_idx-20:current_idx].mean()
            current_bb_width = self.data['bb_width'].iloc[current_idx]
            
            if current_bb_width < bb_width_20d_avg * 0.6:
                score += 6  # 布林带宽度显著收缩
            elif current_bb_width < bb_width_20d_avg * 0.8:
                score += 4  # 布林带宽度适度收缩
                
        # 过去形成的高波动后回落 (可能是价格稳定的信号)
        if current_idx > 5:
            max_rel_atr = self.data['relative_atr'].iloc[current_idx-5:current_idx].max()
            current_rel_atr = self.data['relative_atr'].iloc[current_idx]
            
            if current_rel_atr < max_rel_atr * 0.6:
                score += 4  # 从高波动回落
            
        return min(score, 15)  # 最高15分
    
    def calculate_buy_signal_score(self, idx=-1):
        """
        计算买入信号强度综合得分 (0-100分)
        
        参数:
        idx (int): 要评估的数据点索引，默认为-1(最后一个数据点)
        
        返回:
        float: 0-100之间的得分，分数越高表示买入信号越强
        dict: 各组成部分的得分明细
        """
        if idx == -1:
            idx = len(self.data) - 1
            
        # 确保指标已经计算完毕
        component_scores = {
            'RSI指标': self._score_rsi(idx),                     # 0-20分
            '价格形态': self._score_price_pattern(idx),           # 0-20分
            '成交量分析': self._score_volume(idx),                # 0-15分
            '支撑位分析': self._score_support_resistance(idx),   # 0-15分
            '动量指标': self._score_momentum(idx),               # 0-15分
            '波动率分析': self._score_volatility(idx)            # 0-15分
        }
        
        # 计算总分
        total_score = sum(component_scores.values())
        
        # 信号强度解释
        signal_strength = "无信号"
        if total_score >= 80:
            signal_strength = "极强买入信号"
        elif total_score >= 70:
            signal_strength = "强买入信号"
        elif total_score >= 60:
            signal_strength = "中强买入信号"
        elif total_score >= 50:
            signal_strength = "中等买入信号"
        elif total_score >= 40:
            signal_strength = "弱买入信号"
        elif total_score >= 30:
            signal_strength = "极弱买入信号"
        
        result = {
            'total_score': total_score,
            'signal_strength': signal_strength,
            'component_scores': component_scores,
            'date': self.data['date'].iloc[idx],
            'price': self.data['close'].iloc[idx]
        }
        
        return result
    
    def evaluate_recent_signals(self, days=10):
        """
        评估最近几天的买入信号强度
        
        参数:
        days (int): 要评估的最近天数
        
        返回:
        pandas.DataFrame: 包含最近几天买入信号评分的DataFrame
        """
        results = []
        for i in range(max(0, len(self.data) - days), len(self.data)):
            score_data = self.calculate_buy_signal_score(i)
            results.append({
                'date': score_data['date'],
                'price': score_data['price'],
                'total_score': score_data['total_score'],
                'signal_strength': score_data['signal_strength'],
                **{f"score_{k}": v for k, v in score_data['component_scores'].items()}
            })
            
        return pd.DataFrame(results)