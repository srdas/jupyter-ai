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

print("FinTools: Loading...BEFORE CLASS")

class FinTools(object):
    """
    A collection of financial tools for data retrieval, analysis, and forecasting.
    """
    print("FinTools: Loading...")
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
        # self.pipeline = BaseChronosPipeline.from_pretrained(
        #     "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        #     device_map="cpu",  # use "cpu" for CPU inference, "cuda" for GPU inference
        #     torch_dtype=torch.bfloat16,
        # )

    print("FinTools: Loaded.")

    def run(self):
        """
        Run the pipeline on the data.
        """
        data = self.data
        return data

    def plot_forecast_data(self, series, quantiles, predicton_length=12):
        """
        Plots the financial data.
        """
        forecast_index = range(len(series), len(series) + predicton_length)
        low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]
        plt.figure(figsize=(8, 4))
        plt.plot(series, color="royalblue", label="historical data")
        plt.plot(forecast_index, median, color="tomato", label="median forecast")
        plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
        plt.legend()
        plt.grid()
        plt.show()

    print("FinTools: Loading FinancialDatasetsTools...")

    def forecast_data(self, series: list, prediction_length: int, quantile_levels=[0.1, 0.5, 0.9]):  
        """
        Forecasts future data using the Chronos model.
        """
        quantiles, mean = self.pipeline.predict_quantiles(
            context=torch.tensor(series).unsqueeze(0),
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
        return quantiles, mean
    
    print("FinTools: Loaded FinancialDatasetsTools.")

    


