# �����������ź�����ϵͳ

�����Ŀ�ṩ��һ�����ڶ����Ӽ��������Ĺ�Ʊ�����ź�����ϵͳ��ͨ�����϶��ּ���ָ���������ۺ��Ե������ź�ǿ�����֡�

## �����ص�

- **�������ۺ�����**����϶��ּ���ָ�꣬����0-100�ֵ��ۺ�����
- **�������ݼ���**��֧�ִ�Yahoo Finance��CSV�ļ����Զ���API��������
- **�����ļ���ָ�����**�����÷ḻ�ļ���ָ����㺯��
- **���ӻ�����**���ṩ���ֿ��ӻ����ߣ�ֱ��չʾ�����ź�
- **����Ԥ����**������������ϴ��Ԥ������
- **֧�ֻز�**����������ʷ���ݻز�

## ��װָ��

```bash
pip install trading-signal-scorer
```

���Դ�밲װ��

```bash
git clone https://github.com/yourusername/trading-signal-scorer.git
cd trading-signal-scorer
pip install -e .
```

### ��������

- Python >= 3.7
- numpy
- pandas
- matplotlib
- plotly
- scikit-learn
- yfinance
- ta-lib (��Ҫ�Ȱ�װTA-Lib C��)

## ��������

```python
from trading_signal_scorer.data_loader import load_from_yahoo
from trading_signal_scorer.buy_signal_scorer import BuySignalScorer
from trading_signal_scorer.plot_utils import plot_price_with_signals

# ���ع�Ʊ����
data = load_from_yahoo("AAPL", period="1y")

# ���������������������ź�
scorer = BuySignalScorer(data)
signal = scorer.calculate_buy_signal_score()

# ������ֽ��
print(f"�����ź�ǿ��: {signal['total_score']:.2f}/100")
print(f"�źŽ��: {signal['signal_strength']}")

# ���ƴ��������źŵļ۸�ͼ��
fig = plot_price_with_signals(data, scorer, threshold=50)
fig.savefig("buy_signals.png")
```

## ����ϵͳ˵��

�����ź�������������Ҫ��ɲ��ֹ��ɣ��ܷ�Ϊ100�֣�

1. **RSIָ��**��0-20�֣�������RSI��������͵ױ����������
2. **�۸���̬**��0-20�֣������ڲ��ִ����ƶ�ƽ���ߺ�����ͼ��̬����
3. **�ɽ�������**��0-15�֣���������Գɽ�����OBVָ������
4. **֧��λ����**��0-15�֣������ڼ���֧��λ��쳲������ص�λ����
5. **����ָ��**��0-15�֣����������ָ�ꡢ����ָ���MACD����
6. **�����ʷ���**��0-15�֣�������ATR�Ͳ��ִ��������

���ֱ�׼��
- 80-100�֣���ǿ�����ź�
- 70-79�֣�ǿ�����ź�
- 60-69�֣���ǿ�����ź�
- 50-59�֣��е������ź�
- 40-49�֣��������ź�
- 30-39�֣����������ź�
- 0-29�֣��������ź�

## ʹ��ʾ��

��ϸ��ʹ��ʾ����ο� `examples/` Ŀ¼�µ�ʾ���ļ���

## ���֤

MIT