import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('Advertising.csv')
print(df.head())

print(df.isnull().sum())

# Visualize data
sns.pairplot(df)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# Feature selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Display coefficients
coeffs = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coeffs)

# Visualize actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('predicted_sales.csv', index=False)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"RF R2 Score: {r2_score(y_test, rf_pred)}")
