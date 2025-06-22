import pandas as pd
from stepwise import BidirectionalStepwiseSelection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor

data = pd.read_csv('data/features/simplified_features.csv')

top100= data[data['rank']<=100]

meta = ['appid','title','rank']
keep_cols = [c for c in data.columns if c not in meta]

X = top100[keep_cols]
y = top100['rank']

selected_vars, iteration_log, final_model = BidirectionalStepwiseSelection(
    X,
    y,
    model_type="linear",            # ordinary least squares
    elimination_criteria="aic",      # minimize Akaike Information Criterion
    varchar_process="dummy_dropfirst",
    senter=0.15,                     # p-value threshold to enter
    sstay=0.10                       # p-value threshold to stay
)

print("=== Selected Variables ===")
print(selected_vars)

print("\n=== Selection Trace ===")
print(iteration_log)

print("\n=== Final Model Summary ===")
print(final_model.summary())

# drop the intercept placeholder
selected_vars = [v for v in selected_vars if v != "intercept"]

X = top100[selected_vars]
y=top100['rank']

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

neg_mae = cross_val_score(model, X, y,
                          scoring=mae_scorer,
                          cv=kf,
                          n_jobs=-1)
neg_mse = cross_val_score(model, X, y,
                          scoring=mse_scorer,
                          cv=kf,
                          n_jobs=-1)

mae = -neg_mae.mean()
mse = -neg_mse.mean()
rmse = mse**0.5

print("10-Fold CV Results:")
print(f"  • MAE:  {mae:.3f}")
print(f"  • MSE:  {mse:.3f}")
print(f"  • RMSE: {rmse:.3f}")


dummy = DummyRegressor(strategy="mean")
neg_mae_base = cross_val_score(dummy, X, y,
                               scoring=mae_scorer,
                               cv=kf,
                               n_jobs=-1)
neg_mse_base = cross_val_score(dummy, X, y,
                               scoring=mse_scorer,
                               cv=kf,
                               n_jobs=-1)

baseline_mae  = -neg_mae_base.mean()
baseline_mse  = -neg_mse_base.mean()
baseline_rmse = baseline_mse ** 0.5

# --- print comparison ---
print("\nComparison vs. Mean-Predictor Baseline:")
metrics = {
    "MAE": (mae,  baseline_mae),
    "MSE": (mse,  baseline_mse),
    "RMSE": (rmse, baseline_rmse)
}

for name, (m, b) in metrics.items():
    delta = b - m
    print(f"  • {name:4}: Model = {m:.3f}, Baseline = {b:.3f}, Δ = {delta:+.3f}")