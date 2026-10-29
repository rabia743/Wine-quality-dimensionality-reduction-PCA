import pandas as pd
from sklearn.decomposition import PCA
import joblib


data = pd.read_csv("wine_scaled.csv")

# Keep 95% of the information in the data
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(data)

# Save the reduced data
columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
pca_data = pd.DataFrame(pca_data, columns=columns)
pca_data.to_csv("wine_pca.csv", index=False)

# Save the explained variance for browser visualizations
variance_data = pd.DataFrame(
    {
        "Component": columns,
        "ExplainedVarianceRatio": pca.explained_variance_ratio_,
        "CumulativeExplainedVariance": pca.explained_variance_ratio_.cumsum(),
    }
)
variance_data.to_csv("wine_pca_variance.csv", index=False)

# Save the trained PCA model
joblib.dump(pca, "wine_pca_model.pkl")

print("Original data shape:", data.shape)
print("Reduced data shape:", pca_data.shape)
print("Variance kept:", pca.explained_variance_ratio_.sum())
print("Number of PCA components:", pca.n_components_)