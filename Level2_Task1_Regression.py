import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("sales_dataset.csv")

# Remove rows where important columns are missing
df = df.dropna(subset=['Quantity', 'Unit_Price', 'Total_Sales'])

# Remove duplicates
df = df.drop_duplicates()

# Select only numeric columns for regression
X = df[['Quantity', 'Unit_Price']]
y = df['Total_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
