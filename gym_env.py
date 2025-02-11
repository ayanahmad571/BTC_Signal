import gym
import numpy as np
import pandas as pd
from gym import spaces

class BitcoinTradingEnv(gym.Env):
    def __init__(self, df):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df  # Data containing technical indicators and price info
        self.current_step = -1  # Initialize step counter

        # Action space: 3 discrete actions (Buy, Sell, Hold)
        self.action_space = spaces.Discrete(3)

        # Observation space: All technical indicators as features (except timestamp)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns) - 1,), dtype=np.float32)

        # Set starting balance (1 million USD for adequate trading capability)
        self.balance = 1000000  # Starting with $1,000,000 in balance
        self.holdings = 0  # No Bitcoin holdings initially
        self.total_reward = 0  # Track the total reward over the episode
        

    def reset(self):
        """Reset the environment at the beginning of each episode."""
        self.current_step = 0
        self.balance = 1000000  # Reset balance
        self.holdings = 0  # Reset holdings
        self.total_reward = 0  # Reset total reward
        return self._next_observation()  # Return the initial observation (state)

    def _next_observation(self):
        """Return the current observation (technical indicators/features)."""
        return np.array(self.df.iloc[self.current_step].values[1:], dtype=np.float32)  # Exclude timestamp

    def step(self, action):
        """Take an action (Buy/Sell/Hold) and return the next state, reward, and done flag."""
        current_price = self.df.iloc[self.current_step]["close"]  # Get current Bitcoin price
        print(f"PRICE_FROM_GYM------{current_price}")

        if action == 0:  # Buy action
            if self.balance >= current_price:  # Only buy if balance is enough
                # Buy as much Bitcoin as the available balance allows
                self.holdings += self.balance / current_price  
                self.balance = 0  # All balance is spent to buy Bitcoin
        elif action == 1:  # Sell action
            if self.holdings > 0:  # Only sell if we hold Bitcoin
                self.balance += self.holdings * current_price  # Sell Bitcoin holdings
                self.holdings = 0  # After selling, no holdings left
        # Action 2 (Hold) doesn't affect balance or holdings

        # Calculate reward as the profit/loss compared to initial balance (1 million USD)
        portfolio_value = self.balance + (self.holdings * current_price)
        reward = portfolio_value - 1000000  # Reward is profit/loss compared to $1,000,000 initial
        self.total_reward += reward  # Track total reward

        # Move to the next step (timeframe)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1  # Episode ends if we reach the end of the data

        # Return the observation, reward, done flag, and info with portfolio value
        info = {'portfolio_value': portfolio_value}
        return self._next_observation(), reward, done, info, self.holdings


    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}")
