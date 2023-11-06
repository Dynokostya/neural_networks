import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Fetch the dataset
housing = fetch_california_housing()
X = housing.data
feature_names = housing.feature_names

# StandardScaler for data normalization
scaler = StandardScaler()

# Fit the scaler to the data and transform
X_scaled = scaler.fit_transform(X)

# Plotting the original and scaled data
fig, axes = plt.subplots(2, len(feature_names), figsize=(20, 8))
for i, feature_name in enumerate(feature_names):
    # Original data
    axes[0, i].hist(X[:, i], bins=50, color='blue', alpha=0.7)
    axes[0, i].set_title(feature_name)
    axes[0, i].set_ylabel('Original')

    # Scaled data
    axes[1, i].hist(X_scaled[:, i], bins=50, color='orange', alpha=0.7)
    axes[1, i].set_ylabel('Scaled')

plt.tight_layout()
plt.show()
