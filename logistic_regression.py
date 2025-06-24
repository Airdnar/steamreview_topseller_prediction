import pandas as pd
from stepwise import BidirectionalStepwiseSelection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

# 1. Load and clean data
data = pd.read_csv('data/features/simplified_features.csv')

# 2. Build binary target: 1 if rank ≤ 10 (Top 10), else 0
data['is_top10'] = (data['rank'] <= 10).astype(int)

# 3. Feature filtering: drop columns with ≥80% zeros
meta_cols = ['appid', 'title', 'rank', 'is_top10']
feat_cols = [c for c in data.columns if c not in meta_cols]
zero_frac = (data[feat_cols] == 0).mean(axis=0)
keep_cols = zero_frac[zero_frac < 0.8].index.tolist()

X = data[keep_cols]
y = data['is_top10']

# 4. Stepwise selection (logistic, AIC-based)
selected_vars, iteration_log, final_model = BidirectionalStepwiseSelection(
    X,
    y,
    model_type="logistic",        # logistic regression  
    elimination_criteria="aic",   # minimize AIC
    varchar_process="dummy_dropfirst",
    senter=0.20,                  # p-value to enter
    sstay=0.20                    # p-value to stay
)

print("=== Selected Variables ===")
print(selected_vars)
print("\n=== Selection Trace ===")
print(iteration_log)
print("\n=== Final Model Summary ===")
print(final_model.summary())

# drop the intercept placeholder
selected_vars = [v for v in selected_vars if v != "intercept"]

# 5. Prepare data with only selected features (no resampling)
X_sel = data[selected_vars]
y_sel = data['is_top10']

# 6. 5-fold stratified CV for your model
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

scorers = {
    'Accuracy':  make_scorer(accuracy_score),
    'ROC AUC':   make_scorer(roc_auc_score, needs_proba=True),
    'Precision': make_scorer(precision_score, average = 'macro', zero_division=0),
    'Recall':    make_scorer(recall_score, average = 'macro', zero_division=0),
}

model_results = {}
for name, scorer in scorers.items():
    scores = cross_val_score(clf, X_sel, y_sel, scoring=scorer, cv=cv5, n_jobs=-1)
    model_results[name] = scores.mean()

print("5-Fold CV Results (Logistic Regression):")
for metric, val in model_results.items():
    print(f"  • {metric}: {val:.3f}")

# 7. 5-fold stratified CV for majority‐class baseline
baseline = DummyClassifier(strategy='most_frequent', random_state=42)
baseline_results = {}
for name, scorer in scorers.items():
    scores = cross_val_score(baseline, X_sel, y_sel, scoring=scorer, cv=cv5, n_jobs=-1)
    baseline_results[name] = scores.mean()

# 8. Compare side-by-side
print("\nComparison vs. Majority-Class Baseline:")
for metric in model_results:
    m = model_results[metric]
    b = baseline_results[metric]
    print(f"  • {metric:9}: Model = {m:.3f}, Baseline = {b:.3f}, Δ = {m-b:+.3f}")