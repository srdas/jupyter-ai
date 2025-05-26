# Various finance tools that are useful

print("FinTools: BOOT UP...")

import os
import requests
from .fd import FinancialDatasetsTools

import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline # requires: pip install chronos-forecasting
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np
import yfinance as yf  # requires: pip install yfinance
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from agno.tools import Toolkit
from agno.utils.log import log_error



print("FinTools: Loading...BEFORE CLASS")

class FinTools(Toolkit):
    """
    A collection of financial tools for data retrieval, analysis, and forecasting.
    """
    print("FinTools: Loading...")
    def __init__(
        self,
        enable_arima: bool = True,
    ):
        # Register the tools
        if enable_arima:
            self.register(self.arima_forecast)

    print("FinTools: initialized.")

    # For testing purposes
    def run(self):
        """
        Run the pipeline on the data.
        """
        data = self.data
        return data
    

    def get_financial_data(self, ticker, start_date, end_date):
        """
        Fetches historical financial data from Yahoo Finance for a given ticker symbol.
        May not work because of Yahoo Finance API changes and rate limiting.
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("No data found for the given ticker and date range.")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    

    def plot_forecast(self, series, forecast):
        """
        Plots the financial data.
        """
        forecast_index = range(len(series), len(series) + len(forecast))
        plt.figure(figsize=(8, 4))
        plt.plot(series, color="royalblue", label="historical data")
        plt.plot(forecast_index, forecast, color="tomato", label="forecast")
        plt.legend()
        plt.grid()
        plt.show()

    
    def chronos_forecast(self, series, prediction_length, quantile_levels=[0.1, 0.5, 0.9]):  
        """
        Forecasts future data using the Chronos model.
        """
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map="cpu",  # use "cpu" for CPU inference, "cuda" for GPU inference
            torch_dtype=torch.float32,
        )
        print("FinTools: Chronos ... PIPELINE LOADED")
        quantiles, mean = pipeline.predict_quantiles(
            context=torch.tensor(series, dtype=torch.float32),
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
        return quantiles, mean
    

    def arima_forecast(self, series, prediction_length, p=61, d=0, q=1):
        """
        Forecasts future data using the ARIMA model.
        """
        model = ARIMA(series, order=(p, d, q))
        print("FinTools: ARIMA ... PIPELINE LOADED")
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=prediction_length)
        return forecast
    

    def prophet_forecast(self, df, prediction_length):
        """
        Forecasts future data using the Prophet model.
        df must a dataframe with two columns: 'ds' and 'y'.
        'ds' is the date column and 'y' is the value column.
        """
        model = Prophet()
        print("FinTools: Prophet ... PIPELINE LOADED")
        model.fit(df)
        future = model.make_future_dataframe(periods=prediction_length)
        forecast = model.predict(future)
        return forecast["yhat"][-prediction_length:]


    print("FinTools: Loaded.")

    


