import pandas as pd

# Load the dataset
df = pd.read_csv("btc_hourly_features.csv")

# Check for any NaN values in the dataset
print("Checking for NaN values:")
print(df.isna().sum())  # This will show the count of NaNs for each column

# Check for infinite values
print("\nChecking for infinite values:")
print((df == float('inf')).sum())  # Check for positive infinity
print((df == float('-inf')).sum())  # Check for negative infinity



# Fetch rows where 'volume_change_1h' column contains infinite values
infinite_rows = df[df['volume_change_1h'].isin([float('inf'), float('-inf')])]

# Display the rows with infinite values in 'volume_change_1h'
print(infinite_rows)
