# Spread Option Pricing Simulation

This repository contains Python code to simulate and price spread options using various methods, including **Geometric Brownian Motion (GBM)**, **Correlated GBM**, and **Ornstein-Uhlenbeck (OU)** processes. It also includes Monte Carlo simulations and exponential weighting to compute statistics like volatility and correlation from historical forward prices.

---

## Features

- Simulates forward price paths for two assets using GBM, Correlated GBM, or OU processes.
- Monte Carlo simulation to compute option fair values and variability.
- Incorporates exponential weighting for volatility estimation.
- Supports correlation calculation across multiple quarters.
- Customizable via a JSON configuration file.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib openpyxl

## Usage

1. Prepare the config.json file and data.xlsx file with your desired inputs.
2. Run the script:
   ```bash
    python main.py
3.  The script outputs the fair value of the spread option for each simulation method.
