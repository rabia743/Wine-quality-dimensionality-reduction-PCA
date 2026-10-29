from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

data = pd.read_csv("cleaned wine data.csv")

# Select 11 features
X = data.drop(columns=["quality"])

# Apply feature scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(X)

# Save scaler
# Convert scaled array into DataFrame
scaled_data = pd.DataFrame(scaled, columns=X.columns)
# Save scaled data
scaled_data.to_csv("wine_scaled.csv", index=False)
joblib.dump(scaler, "wine_scaler.pkl")

print(scaled[:3])
print("Shape:", scaled.shape)
