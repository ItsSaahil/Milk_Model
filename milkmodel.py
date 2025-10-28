import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and prepare the CSV file
print("Loading and preparing data...")
df = pd.read_csv('dummy_data_300.csv')

# Convert Timestamp to datetime and extract hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# Clean up column names
df.columns = [col.strip() for col in df.columns]

# Step 2: Analyze the data
print("\nData Overview:")
print(df.describe())

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Step 3: Define features (X) and target (y)
# For this example, let's predict Water Content based on other features
X = df[['Temperature (°C)', 'pH Level', 'Light Intensity', 'Hour']]
y = df['Water Content']

print("\nFeature columns:", X.columns.tolist())
print("Number of samples:", len(df))

# Step 4: Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Root Mean Squared Error: {rmse:.6f}")
print(f"R² Score: {r2:.6f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Step 7: Make predictions for example data points
print("\nMaking predictions for example data points...")
example_data = pd.DataFrame({
    'Temperature (°C)': [21.0, 23.0],
    'pH Level': [7.8, 7.6],
    'Light Intensity': [200, 210],
    'Hour': [12, 15]
})

predictions = model.predict(example_data)
print("\nExample Predictions:")
for i, pred in enumerate(predictions):
    print(f"\nScenario {i+1}:")
    print(f"Input Parameters:")
    for col in example_data.columns:
        print(f"  {col}: {example_data.iloc[i][col]}")
    print(f"Predicted Water Content: {pred:.4f}")

# Step 8: Save the model and feature information
print("\nSaving model and feature information...")
model_info = {
    'model': model,
    'feature_names': X.columns.tolist(),
    'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}
joblib.dump(model_info, 'random_forest_model.pkl')
print("Model and feature information saved as 'random_forest_model.pkl'")