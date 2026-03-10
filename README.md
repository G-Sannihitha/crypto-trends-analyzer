# Crypto Trends Analyzer — Bitcoin Price Forecasting

A machine learning pipeline to forecast Bitcoin price direction using technical indicators, 
time-series models, and Wikipedia-based sentiment signals.

## Results
- **ROC-AUC: 0.72** on directional price classification
- **61% out-of-sample accuracy** tested on unseen data (2022–2025)
- Sentiment features boosted accuracy **0.58 → 0.63** (confirmed via feature ablation study)

## Tech Stack
Python, Scikit-learn, XGBoost, TensorFlow/Keras, Prophet, Pandas, yfinance, CoinGecko API

## Models Compared
LSTM, Prophet, XGBoost, Random Forest, Gradient Boosting, AdaBoost

## Features Engineered
RSI, SMA20/SMA50, rolling volatility (10-day STD), daily price change %,
Wikipedia edit-frequency and sentiment score as external signal

## Repository Structure
```
├── data/
│   ├── btc.csv                 # Historical Bitcoin price data
│   └── wikipedia_edits.csv     # Wikipedia edits time series
├── LiveCryptoForecast
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook LiveCryptoForecast.ipynb
```

Run all cells in order. The notebook will load data, train all models, and output 
performance metrics, forecast plots, and confusion matrices.

## Data Preparation
1. Create a `/data` directory in the project root
2. Place `btc.csv` and `wikipedia_edits.csv` inside `/data`
3. Update file paths in the notebook if needed:
```python
file_path = "data/btc.csv"
wiki_path = "data/wikipedia_edits.csv"
```
