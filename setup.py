### 3. setup.py

```python
from setuptools import setup, find_packages

setup(
    name="trading_signal_scorer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="T型交易低点识别系统",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trading-signal-scorer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "ta-lib>=0.4.0",
        "plotly>=5.3.0",
        "tqdm>=4.62.0",
        "yfinance>=0.1.70",
    ],
)