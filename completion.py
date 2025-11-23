import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBClassifier
import openml
from scipy.stats import spearmanr

# Load dataset from OpenML
dataset = openml.datasets.get_dataset(46805)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute)

# Convert target to numeric
y = pd.to_numeric(y)

print(f"Dataset shape: {X.shape}")
print(f"Target name: {dataset.default_target_attribute}")
print(f"Target class distribution:\n{pd.Series(y).value_counts()}")

# Convert the categorical feature to one-hot encoding
X = pd.get_dummies(X, columns=['Highest_degree_awarded'], drop_first=True)

# Save preprocessed dataset (optional)
data = X.copy()
data['target'] = y
data.to_csv('data.csv', index=False)

# Feature selection methods
results = {}

# 1. Random Forest Feature Selection
def rf_feature_selection(X, y, k=10):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:k]
    return indices, feature_importances

# 2. XGBoost Feature Selection
def xgb_feature_selection(X, y, k=10):
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X, y)
    feature_importances = xgb.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:k]
    return indices, feature_importances

# 3. Logistic Regression Feature Selection
def lr_feature_selection(X, y, k=10):
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)
    feature_importances = np.abs(lr.coef_[0])
    indices = np.argsort(feature_importances)[::-1][:k]
    return indices, feature_importances

# 4. Feature Agglomeration
def fa_feature_selection(X, y, k=10):
    # Use more clusters than needed to have more options
    n_clusters = min(50, X.shape[1] // 2)
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    
    # Calculate feature importance for each feature based on variance
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        feature_importances[i] = np.var(X.iloc[:, i])
    
    # Create a list that stores (feature_index, cluster_label, importance)
    feature_data = []
    for i in range(X.shape[1]):
        feature_data.append((i, agglo.labels_[i], feature_importances[i]))
    
    # Sort by importance (descending)
    feature_data.sort(key=lambda x: x[2], reverse=True)
    
    # Take top k features across all clusters
    selected_indices = [f[0] for f in feature_data[:k]]
    
    return selected_indices, feature_importances

# 5. Highly Variable Gene Selection
def hvgs_feature_selection(X, y, k=10):
    variances = np.var(X, axis=0)
    indices = np.argsort(variances)[::-1][:k]
    return indices, variances

# 6. Spearman's Correlation
def spearman_feature_selection(X, y, k=10):
    correlations = []
    
    for i in range(X.shape[1]):
        corr, _ = spearmanr(X.iloc[:, i], y)
        correlations.append(abs(corr))
    
    correlations = np.array(correlations)
    indices = np.argsort(correlations)[::-1][:k]
    return indices, correlations

# Function to get feature names
def get_feature_names(X):
    return X.columns.tolist()

# Function to run cross-validation with selected features
def run_cv(X, y, selected_indices, method_name):
    X_selected = X.iloc[:, selected_indices]
    
    if method_name in ['Random Forest', 'FA', 'HVGS', 'Spearman']:
        clf = RandomForestClassifier(random_state=42)
    elif method_name == 'XGBoost':
        clf = XGBClassifier(random_state=42)
    else:  # Logistic Regression
        clf = LogisticRegression(random_state=42, max_iter=1000)
    
    scores = cross_val_score(clf, X_selected, y, cv=10, scoring='accuracy')
    return scores.mean(), scores.std()

# Create a results dictionary to store all the information
results = {
    'method': [],
    'CV10': [],
    'CV9': [],
    'top10_features': [],
    'top9_features': []
}

# Dictionary mapping method names to feature selection functions
method_functions = {
    'Random Forest': rf_feature_selection,
    'XGBoost': xgb_feature_selection,
    'Logistic Regression': lr_feature_selection,
    'FA': fa_feature_selection,
    'HVGS': hvgs_feature_selection,
    'Spearman': spearman_feature_selection
}

# Run each method
for method_name, selection_func in method_functions.items():
    print(f"\nRunning {method_name}...")
    
    # Set 1: Top 10 features
    selected_indices, feature_importances = selection_func(X, y, k=10)
    feature_names = get_feature_names(X)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Cross-validation with top 10 features
    cv10_mean, cv10_std = run_cv(X, y, selected_indices, method_name)
    print(f"{method_name} CV with top 10 features: {cv10_mean:.4f} ± {cv10_std:.4f}")
    
    # Remove the highest ranked feature
    top_feature_idx = selected_indices[0]
    X_reduced = X.drop(columns=[feature_names[top_feature_idx]])
    
    # Set 2: Top 9 features from reduced dataset
    selected_indices_reduced, _ = selection_func(X_reduced, y, k=9)
    feature_names_reduced = get_feature_names(X_reduced)
    selected_features_reduced = [feature_names_reduced[i] for i in selected_indices_reduced]
    
    # Map back to original indices for cross-validation
    original_indices_reduced = []
    for idx in selected_indices_reduced:
        col_name = feature_names_reduced[idx]
        original_idx = feature_names.index(col_name)
        original_indices_reduced.append(original_idx)
    
    # Cross-validation with top 9 features
    cv9_mean, cv9_std = run_cv(X, y, original_indices_reduced, method_name)
    print(f"{method_name} CV with top 9 features: {cv9_mean:.4f} ± {cv9_std:.4f}")
    
    # Store results with proper ± symbol
    results['method'].append(method_name)
    # Use the correct ± symbol instead of ﾂｱ
    results['CV10'].append(f"{cv10_mean:.4f} ± {cv10_std:.4f}")
    results['CV9'].append(f"{cv9_mean:.4f} ± {cv9_std:.4f}")
    results['top10_features'].append(", ".join(selected_features[:5]))
    results['top9_features'].append(", ".join(selected_features_reduced[:4]))

# Create and save results table
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)
results_df.to_csv('result.csv', index=False, encoding='utf-8-sig')
