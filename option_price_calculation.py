import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

with open("config.json", "r") as file:
    config = json.load(file)

data_path = config["data_path"]

forward_prices_a = pd.read_excel(data_path, sheet_name=0)  # Load the first sheet --> Country A
forward_prices_b = pd.read_excel(data_path, sheet_name=1)  # Load the second sheet--> Country B


### CONFIG PARAMETERS ###
alpha = config["alpha"] # Exponential decay factor
drift = config["drift"] # Deterministic drift of GBM simulation
num_simulations = config["num_simulations"] # Number of MC simulations
quarters = config["quarters"]  # List of quarters
days_list = config["days_list"]  # List of days corresponding to each quarter
theta = config["theta"] # Speed of mean reversion
discount_rate = config["discount_rate"] # Discount rate of future simulations to NPV
#########################

def exponential_weighting(data, quarter, alpha):
    """
    Calculate the weighted mean and variance using exponential decay weights.

    Args:
        data (pd.DataFrame): The dataframe containing the forward prices.
        quarter (str): The column for which to calculate stats (Q1, Q2, etc.).
        alpha (float): Decay factor controlling the weighting.

    Returns:
        Weighted mean and weighted standard deviation.
    """

    # Reverse the data to give more weight to recent observations
    data = data.iloc[::-1]
    
    # Compute time-based weights (exponential decay)
    weights = np.exp(-alpha * np.arange(len(data)))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    
    # Calculate weighted mean
    mean = np.sum(weights * data[quarter])
    
    # Calculate weighted variance
    variance = np.sum(weights * (data[quarter] - mean) ** 2)
    
    return mean, np.sqrt(variance)

def calculate_correlation_by_quarters(forward_prices_a, forward_prices_b, quarters):
    """
    Calculate the correlation of log returns between two forward price series for each quarter.

    Args:
        forward_prices_a (pd.DataFrame): Forward prices for Country A.
        forward_prices_b (pd.DataFrame): Forward prices for Country B.
        quarters (list): List of quarter columns to analyze (e.g., ['Q1', 'Q2', 'Q3', 'Q4']).

    Returns:
        float: Aggregated Pearson correlation coefficient across all quarters.
    """
    all_returns_a = []
    all_returns_b = []

    for quarter in quarters:
        # Extract prices for the current quarter
        prices_a = forward_prices_a[quarter].dropna()
        prices_b = forward_prices_b[quarter].dropna()

        # Ensure the lengths match
        min_length = min(len(prices_a), len(prices_b))
        prices_a = prices_a.iloc[:min_length]
        prices_b = prices_b.iloc[:min_length]

        # Compute log returns
        returns_a = np.log(prices_a / prices_a.shift(1)).dropna()
        returns_b = np.log(prices_b / prices_b.shift(1)).dropna()

        # Collect returns for correlation calculation
        all_returns_a.extend(returns_a)
        all_returns_b.extend(returns_b)

    # Calculate the overall correlation
    rho = np.corrcoef(all_returns_a, all_returns_b)[0, 1]
    return rho



def simulate_GBM(start_price, mean_volatility, days, drift=0):
    """
    Simulates a Geometric Brownian Motion (GBM) price path.

    Geometric Brownian Motion is widely used in financial modeling to represent the 
    evolution of stock prices or other financial instruments over time.

    Args:
        start_price (float): The initial price of the asset.
        mean_volatility (float): The average volatility of the asset (standard deviation of returns).
        days (int): The number of time steps (days) to simulate.
        drift (float, optional): The expected daily return (drift term). Defaults to 0.

    Returns:
        np.ndarray: An array containing the simulated price path of the asset.
    """

    dt = 1/days
    drift_term = (drift - 0.5 * mean_volatility**2) * dt
    shock_term = mean_volatility*np.sqrt(dt)*np.random.normal(0,1,days)
    daily_returns = drift_term + shock_term
    price_path = start_price*np.exp(np.cumsum(daily_returns))

    return price_path

def simulate_correlated_GBM(start_price_a, start_price_b, vol_a, vol_b, days, drift_a=0, drift_b=0, rho=0):
    """
    Simulate correlated GBM price paths for two assets.

    Args:
        start_price_a (float): Initial price for asset A.
        start_price_b (float): Initial price for asset B.
        vol_a (float): Volatility of asset A.
        vol_b (float): Volatility of asset B.
        days (int): Number of time steps to simulate.
        drift_a (float): Drift for asset A.
        drift_b (float): Drift for asset B.
        rho (float): Correlation coefficient between shocks (-1 to 1).

    Returns:
        np.ndarray, np.ndarray: Simulated price paths for asset A and asset B.
    """
    dt = 1 / days
    prices_a = np.zeros(days)
    prices_b = np.zeros(days)
    
    # Initial prices
    prices_a[0] = start_price_a
    prices_b[0] = start_price_b
    
    # Generate independent random shocks
    Z_a = np.random.normal(0, 1, days)
    Z_b = np.random.normal(0, 1, days)
    
    # Correlate the shocks
    shocks_a = Z_a
    shocks_b = rho * Z_a + np.sqrt(1 - rho**2) * Z_b
    
    # Simulate GBM paths
    for t in range(1, days):
        prices_a[t] = prices_a[t-1] * np.exp((drift_a - 0.5 * vol_a**2) * dt + vol_a * np.sqrt(dt) * shocks_a[t])
        prices_b[t] = prices_b[t-1] * np.exp((drift_b - 0.5 * vol_b**2) * dt + vol_b * np.sqrt(dt) * shocks_b[t])

    
    return prices_a, prices_b

def simulate_OU(start_price, mean_price, mean_reversion_speed, volatility, days):
    """
    Simulate an Ornstein-Uhlenbeck (OU) mean-reverting price path.
    """
    dt = 1 / days
    price_path = np.zeros(days, dtype=float)  # Ensure numeric array
    price_path[0] = float(start_price)  # Ensure numeric start price

    mean_price = float(mean_price)  # Ensure mean_price is numeric

    for t in range(1, days):
        drift = mean_reversion_speed * (mean_price - price_path[t-1]) * dt
        shock = volatility * np.sqrt(dt) * np.random.normal(0, 1)
        price_path[t] = price_path[t-1] + drift + shock

    return price_path



def MC_simulation(num_simulations, forward_prices_a, forward_prices_b, alpha, drift, quarters, days_list, simulate_func, **kwargs):
    """
    Perform Monte Carlo simulation with flexible simulation methods: GBM, correlated GBM, or OU.

    Args:
        simulate_func: Function used for simulation (GBM, correlated GBM, or OU).
        **kwargs: Additional parameters for specific simulation methods (e.g., rho, theta).
    """
    all_spreads = {quarter: [] for quarter in quarters}

    for simulation in range(num_simulations):
        for quarter, days in zip(quarters, days_list):
            # Country A parameters
            start_price_a = exponential_weighting(forward_prices_a, quarter, alpha)[0]
            std_a = exponential_weighting(forward_prices_a, quarter, alpha)[1] / start_price_a
            
            # Country B parameters
            start_price_b = exponential_weighting(forward_prices_b, quarter, alpha)[0]
            std_b = exponential_weighting(forward_prices_b, quarter, alpha)[1] / start_price_b

            # Check simulation method
            if simulate_func == simulate_correlated_GBM:
                daily_prices_a, daily_prices_b = simulate_func(
                    start_price_a=start_price_a,
                    start_price_b=start_price_b,
                    vol_a=std_a,
                    vol_b=std_b,
                    days=days,
                    drift_a=drift,
                    drift_b=drift,
                    rho=kwargs.get('rho', 0)
                )
            elif simulate_func == simulate_OU:
                theta = kwargs.get('theta', 0.1)  # Mean-reversion speed
                mean_price_a = float(kwargs.get('mean_price_a', forward_prices_a[quarter].mean()))  # Use .mean() if it's a Series
                mean_price_b = float(kwargs.get('mean_price_b', forward_prices_b[quarter].mean()))  # Ensure a single float value


                daily_prices_a = simulate_func(start_price_a, mean_price_a, theta, std_a, days)
                daily_prices_b = simulate_func(start_price_b, mean_price_b, theta, std_b, days)
            else:
                # Default to independent GBM
                daily_prices_a = simulate_func(start_price_a, mean_volatility=std_a, days=days, drift=drift)
                daily_prices_b = simulate_func(start_price_b, mean_volatility=std_b, days=days, drift=drift)

            # Calculate daily spread
            daily_spread = daily_prices_b - daily_prices_a
            all_spreads[quarter].append(daily_spread)

    return all_spreads


def option_price(spreads, days_list):
    """
    Compute the option price based on Monte Carlo simulated spreads.

    Args:
        spreads (dict): A dictionary where each key represents a quarter (e.g., 'Q1', 'Q2') 
                        and the value is a list of simulated spreads for each Monte Carlo path.
        discount_rate (float): The risk-free discount rate.
        days_list (list): A list of integers representing the number of days in each quarter.

    Returns:
        tuple: (option_value, price_variability)
               - option_value: The mean of discounted payoffs (i.e., the option price).
               - price_variability: The standard deviation of discounted payoffs.
    """
    all_discounted_payoffs = []  # Collect discounted payoffs across all paths

    for quarter, days in zip(spreads.keys(), days_list):
        # Calculate payoffs for each path in the quarter
        payoffs = np.maximum(spreads[quarter], 0)  # Apply the payoff formula: max(Spread, 0)
        
        # Calculate discount factors for all days in the quarter
        discount_factors = np.exp(-discount_rate * np.arange(days) / days)
        
        # Discount the payoff for each path
        discounted_payoffs = np.sum(payoffs * discount_factors, axis=1)
        
        # Collect all payoffs for this quarter
        all_discounted_payoffs.extend(discounted_payoffs)

    # Compute the mean (option price) and standard deviation (variability)
    option_value = np.mean(all_discounted_payoffs)
    var = np.std(all_discounted_payoffs)

    return option_value,var


def run_simulation(simulator_func, **kwargs):
    """
    Run the Monte Carlo simulation using the specified simulation function.

    Args:
        simulator_func (function): The simulation function to use (GBM, correlated GBM, or OU).
        **kwargs: Additional parameters specific to the simulation function.

    Returns:
        dict: Average spreads calculated for each quarter.
    """
    return MC_simulation(
        num_simulations=num_simulations,
        forward_prices_a=forward_prices_a,
        forward_prices_b=forward_prices_b,
        alpha=alpha,
        drift=drift,
        quarters=quarters,
        days_list=days_list,
        simulate_func=simulator_func,
        **kwargs
    )

# Filter forward prices to start January 2024

forward_prices_a = forward_prices_a[forward_prices_a['Exchange Date'] >= '2024-01-01']
forward_prices_b = forward_prices_b[forward_prices_b['Exchange Date'] >= '2024-01-01']

# Calculate correlation for correlated GBM


rho = calculate_correlation_by_quarters(forward_prices_a, forward_prices_b, quarters)

for simulator_func in [simulate_OU, simulate_GBM, simulate_correlated_GBM]:
    # Define method-specific parameters
    kwargs = {}
    if simulator_func == simulate_correlated_GBM:
        kwargs["rho"] = rho
    elif simulator_func == simulate_OU:
        kwargs["theta"] = theta  # Mean-reversion speed

    # Run the simulation
    spread_to_price = run_simulation(simulator_func, **kwargs)
    option_fair_value,var= option_price(spread_to_price,days_list)
    print(f"The fair value of the spread option using function {simulator_func.__name__} is: {option_fair_value:.2f} ")


