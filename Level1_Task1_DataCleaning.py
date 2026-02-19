import pandas as pd

# Load dataset
df = pd.read_csv("sales_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data
df.to_csv("sales_dataset.csv", index=False)

print("\nData Cleaning Completed!")
