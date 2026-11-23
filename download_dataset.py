from datasets import load_dataset
import pandas as pd

# Download dataset
dataset = load_dataset(
    "hesandism/colombo-tea-auction-prices"
)

# Convert to DataFrame
df = dataset["train"].to_pandas()

# Save CSV file
df.to_csv(
    "colombo_tea_auction.csv",
    index=False
)

print("Dataset downloaded successfully!")

print("Dataset Shape:", df.shape)

print(df.head())