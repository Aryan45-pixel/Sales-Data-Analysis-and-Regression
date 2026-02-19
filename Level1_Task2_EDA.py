import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sales_dataset.csv")

print(df.describe())

# Histogram
df.hist(figsize=(8,6))
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.show()
