import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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

# 5. Define classifiers (no oversampling)
tree_clf  = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5)
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)

# 6. Use built-in scoring shortcuts
scoring = {
    'Accuracy' : 'accuracy',
    'Precision': 'precision_macro',
    'Recall'   : 'recall_macro',
    'ROC AUC'  : 'roc_auc'
}

# 7. Cross-validate the Decision Tree
tree_results = {}
for metric_name, sc in scoring.items():
    scores = cross_val_score(
        tree_clf,
        X, y,
        scoring=sc,
        cv=cv,
        n_jobs=-1
    )
    tree_results[metric_name] = scores.mean()

print("5-Fold CV Results (Decision Tree):")
for metric, val in tree_results.items():
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
    'Model':    tree_results,
    'Baseline': dummy_results
})
comparison['Δ'] = comparison['Model'] - comparison['Baseline']

print("\nMetric comparison:")
print(comparison.to_string(float_format='%.3f'))
