import gym
from stable_baselines3 import PPO
from gym_env import BitcoinTradingEnv  # Import the custom environment from gym_env.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained PPO model
model = PPO.load("btc_trading_ppo")

# Load the dataset again
df = pd.read_csv("./new_test_data/btc_hourly_last_n_hours_features.csv")

# Create the environment
env = BitcoinTradingEnv(df)

# Initialize variables
obs = env.reset()
done = False
portfolio_value = [1000000]  # Track portfolio value at each timestep, starting with $1,000,000
hold_value = [1000000]  # Start with the same value for "buy and hold"
holdings_at_action = []  # Track holdings at each action point to shade background

# Get the initial price and calculate initial Bitcoin holdings (if you bought at the start)
initial_price = df.iloc[0]["close"]
initial_balance = 1000000  # Assuming starting balance is $1,000,000
bitcoins_bought = initial_balance / initial_price  # How many BTC you could buy initially

# Test the trained model
while not done:
    action, _ = model.predict(obs)  # Predict the action (Buy/Sell/Hold)
    obs, reward, done, info, holdings = env.step(action)  # Execute the action in the environment
    
    # Track portfolio value from the environment info
    portfolio_value.append(info['portfolio_value'])  # Track PPO model's portfolio value
    
    # Track holdings at each action point
    holdings_at_action.append(holdings)  # Track holdings at each timestep
    
    # Calculate the "hold" value by multiplying the number of Bitcoins bought by the current price
    current_price = df.iloc[env.current_step-1]["close"]
    hold_value.append(bitcoins_bought * current_price)  # Track the "buy and hold" portfolio value
    
    # Optionally, print the portfolio value and actions
    print(f"Action: {action}, Portfolio Value: {info['portfolio_value']}, Hold Value: {hold_value[-1]}")
    
    # Optionally, render the environment (balance, holdings, etc.)
    env.render()

# Plot the portfolio value and the "buy and hold" value over time
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label="Portfolio Value (PPO)", color='b')
plt.plot(hold_value, label="Hold Value (Buy & Hold)", color='r', linestyle='--')

# Highlight the sections where holdings > 0 (Bitcoin held)
for i in range(1, len(holdings_at_action)):
    if holdings_at_action[i] > 0:  # If holdings are greater than 0, we shade the area
        plt.axvspan(i+1, i+2, color='yellow', alpha=0.3)  # Shaded region for holding Bitcoin

# Title and labels
plt.title("Portfolio Value vs Buy & Hold Value Over Time (Test Phase)")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.show()
