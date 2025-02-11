# train.py
import gym
from stable_baselines3 import PPO
from gym_env import BitcoinTradingEnv  # Import the custom environment from gym_env.py
import pandas as pd

# Load your data (with technical indicators)
df = pd.read_csv("btc_hourly_features.csv")

# Create the custom environment
env = BitcoinTradingEnv(df)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model (adjust the total_timesteps for more training)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("btc_trading_ppo")
print("✅ PPO training complete!")
