# Crypto Trends Analyzer — Bitcoin Price Forecasting
# Full pipeline: data collection, feature engineering, model training & evaluation
# See LiveCryptoForecast.ipynb for complete implementation

import pandas as pd
import numpy as np
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from prophet import Prophet
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Entry point — run full pipeline from notebook
if __name__ == "__main__":
    print("Run LiveCryptoForecast.ipynb for the full pipeline.")
