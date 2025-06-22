import pandas as pd
import numpy as np
from collections import defaultdict


df = pd.read_csv('data/features/all_games_multilang_features.csv')
df.fillna(0, inplace=True)

# Identify language-specific prefixes (excluding identifier columns)
meta = ['rank', 'appid', 'title']
prefixes = [col.split('_')[0] for col in df.columns if '_' in col and col not in meta]

# Group columns by their metric name (everything after the first underscore)
metric_cols = defaultdict(list)
for col in df.columns:
    if '_' in col:
        prefix, metric = col.split('_', 1)
        if prefix in prefixes:
            metric_cols[metric].append(col)
            
exclude = [m for m in metric_cols
              if 'length' in m or 'topic' in m]

# Compute the row-wise average for each metric across available languages, ignoring NaNs
for metric, cols in metric_cols.items():
    if metric in exclude:
        continue
    df[f'{metric}_avg'] = df[cols].mean(axis=1, skipna=True)

# Sort the average columns for consistent ordering
avg_cols = sorted([c for c in df.columns if c.endswith('_avg')])

exclude_cols = [col for m in exclude for col in metric_cols[m]]
info = meta + avg_cols + exclude_cols

compact = df[info]
compact.to_csv('data/features/compact_features.csv', index=False)

variables = [c for c in compact.columns if c not in meta]

corr_matrix = compact[variables].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
to_drop

trimmed = compact.drop(columns=to_drop)

trimmed.to_csv('data/features/trimmed_features.csv', index=False)

df = pd.read_csv('data/features/all_games_multilang_features.csv')
df.fillna(0, inplace=True)
meta = ['appid','title','rank']
variables = [c for c in df.columns if c not in meta]

X = df[variables]
y = df['rank']

corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
to_drop

df.drop(to_drop, axis = 1, inplace=True)
df.to_csv('data/features/simplified_features.csv', index=False)



