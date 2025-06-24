import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

# 1. Load and clean data
data = pd.read_csv('data/features/simplified_features.csv')

# 2. Build binary target: 1 if rank ≤ 10 (Top 10), else 0
data['is_top10'] = (data['rank'] <= 10).astype(int)

# 3. Feature filtering: drop columns with ≥80% zeros
meta_cols = ['appid', 'title', 'rank', 'is_top10']
keep_cols = [c for c in data.columns if c not in meta_cols]

X = data[keep_cols]
y = data['is_top10']

# 4. Set up 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Define classifiers
rf_clf    = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=9,
    random_state=42
)

dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)

# 6. Use built-in scoring shortcuts
scoring = {
    'Accuracy' : 'accuracy',
    'Precision': 'precision_macro',
    'Recall'   : 'recall_macro',
    'ROC AUC'  : 'roc_auc'
}

# 7. Cross-validate the Random Forest
rf_results = {}
for metric_name, sc in scoring.items():
    scores = cross_val_score(
        rf_clf,
        X, y,
        scoring=sc,
        cv=cv,
        n_jobs=-1
    )
    rf_results[metric_name] = scores.mean()

print("5-Fold CV Results (Random Forest):")
for metric, val in rf_results.items():
    print(f"  • {metric}: {val:.3f}")

# 8. Cross-validate the DummyClassifier
dummy_results = {}
for metric_name, sc in scoring.items():
    scores = cross_val_score(
        dummy_clf,
        X, y,
        scoring=sc,
        cv=cv,
        n_jobs=-1
    )
    dummy_results[metric_name] = scores.mean()

# 9. Δ comparison table
comparison = pd.DataFrame({
    'RandomForest': rf_results,
    'Baseline'    : dummy_results
})
comparison['Δ'] = comparison['RandomForest'] - comparison['Baseline']

print("\nMetric comparison:")
print(comparison.to_string(float_format='%.3f'))

# 10. Fit the Random Forest on the entire dataset
rf_clf.fit(X, y)

# 11. Get feature importances and select top 5
importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=keep_cols)
top5 = feat_imp.sort_values(ascending=False).head(5)

# 12. Plot bar chart of top-5 importances
plt.figure(figsize=(8, 6))
plt.barh(top5.index[::-1], top5.values[::-1])  # reverse for descending top→bottom
plt.xlabel("Mean Decrease in Impurity")
plt.title("Top 5 Features by Importance (Random Forest)")
plt.tight_layout()
plt.show()