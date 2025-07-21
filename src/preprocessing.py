import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import joblib
from sklearn.model_selection import GridSearchCV

def preprocess_data(df):
    """
    Preprocess the DataFrame by removing unnecessary columns and handling missing values.
    """
    if 'EJ' in df.columns:
        df['EJ_enc'] = df['EJ'].map({'A': 0, 'B': 1})
        df.drop(columns=['EJ'], inplace=True)
    if 'Target' in df.columns:
        # df['target_labels_enc'] = df['Target'].map({'no_fraud': 0, 'payment_fraud': 1, 'identification_fraud': 2,'malware_fraud':3})
        df['target_enc'] = df['Target'].apply(lambda x: 0 if x == 'no_fraud' else 1)
        df.drop(columns=['Target'], inplace=True)
        # df.drop(columns=['target_labels_enc'], inplace=True)

    # Drop columns that are not needed for analysis
    df = df.drop(columns=['Id'], errors='ignore')
    
    # Fill missing values with the median for numeric columns due to their skewness
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    return df

def correlation_analysis(df, type='pearson'):
    """
    Perform correlation analysis on the DataFrame.
    
    Parameters:
    - df: DataFrame to analyze
    - type: Type of correlation to compute ('pearson', 'spearman', 'kendall')
    
    Returns:
    - corr_matrix: Correlation matrix
    """
    spearman_results = []
    kendall_results = []
    
    if type == 'spearman':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            vals = df[[col, 'target_enc']]
            sp_corr, _ = spearmanr(vals[col], vals['target_enc'])
            spearman_results.append((col, sp_corr))
            spearman_df = pd.DataFrame(spearman_results, columns=['Feature', 'Spearman']).set_index('Feature')
    elif type == 'kendall':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            vals = df[[col, 'target_enc']]
            kd_corr, _ = kendalltau(vals[col], vals['target_enc'])
            kendall_results.append((col, kd_corr))
            kendall_df = pd.DataFrame(kendall_results, columns=['Feature', 'Kendall']).set_index('Feature')
    else:
        raise ValueError("Unsupported correlation type. Use 'pearson', 'spearman', or 'kendall'.")

    if type == 'spearman':
        return spearman_df
    elif type == 'kendall':
        return kendall_df

#TODO: Check how it works
def distance_correlation(x, y):
    """ Compute distance correlation between two arrays"""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    n = x.shape[0]
    a = np.abs(x[:, None] - x)
    b = np.abs(y[:, None] - y)
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).sum() / (n**2)
    dvarx = (A * A).sum() / (n**2)
    dvary = (B * B).sum() / (n**2)
    return 0 if dvarx == 0 or dvary == 0 else np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx * dvary))

def rank_and_aggregate_features(df, n_top=10, run_gridsearch=False, best_rf_params_path='best_rf_params.pkl'):
    """  
    Rank features based on different correlation metrics and aggregate results.
    It results in voting for the most important features.
    """
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['target_labels_enc', 'target_enc']]
    X = df[feature_cols]
    y = df['target_enc']

    # Spearman ranks
    spearman_scores = [(col, abs(spearmanr(df[col], y)[0])) for col in feature_cols]
    spearman_top = set([x[0] for x in sorted(spearman_scores, key=lambda x: x[1], reverse=True)[:n_top]])

    # Kendall ranks
    kendall_scores = [(col, abs(kendalltau(df[col], y)[0])) for col in feature_cols]
    kendall_top = set([x[0] for x in sorted(kendall_scores, key=lambda x: x[1], reverse=True)[:n_top]])

    # Distance correlation
    distcorr_scores = [(col, abs(distance_correlation(df[col].values, y.values))) for col in feature_cols]
    distcorr_top = set([x[0] for x in sorted(distcorr_scores, key=lambda x: x[1], reverse=True)[:n_top]])

    # Random Forest importance with GridSearchCV
    # Random Forest importance
    if run_gridsearch:
        print("Running GridSearchCV for Random Forest (this may take time)...")
        param_grid = {
            'n_estimators': [100, 250, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 0.5, None],
            'class_weight': [None, 'balanced']
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X, y)
        rf_best_params = grid_search.best_params_
        print("Best parameters for Random Forest:", rf_best_params)
        joblib.dump(rf_best_params, 'data/best_rf_params.pkl')
    else:
        try:
            rf_best_params = joblib.load(best_rf_params_path)
        except FileNotFoundError:
            print("No tuned RF params found; using defaults.")
            rf_best_params = {'n_estimators': 100, 'random_state': 42}

    clf = RandomForestClassifier(**rf_best_params, random_state=42)
    clf.fit(X, y)
    rf_scores = list(zip(feature_cols, clf.feature_importances_))
    rf_top = set([x[0] for x in sorted(rf_scores, key=lambda x: x[1], reverse=True)[:n_top]])
    

    # Aggregate: count votes for each feature
    all_top_features = sorted(spearman_top.union(kendall_top, distcorr_top, rf_top))
    votes = Counter()
    for f in all_top_features:
        votes[f] += int(f in spearman_top)
        votes[f] += int(f in kendall_top)
        votes[f] += int(f in distcorr_top)
        votes[f] += int(f in rf_top)

    # Create DataFrame summary
    summary = pd.DataFrame({
        'Votes': [votes[f] for f in all_top_features],
        'Spearman': [dict(spearman_scores).get(f, 0) for f in all_top_features],
        'Kendall': [dict(kendall_scores).get(f, 0) for f in all_top_features],
        'DistCorr': [dict(distcorr_scores).get(f, 0) for f in all_top_features],
        'RF_Importance': [dict(rf_scores).get(f, 0) for f in all_top_features],
    }, index=all_top_features)
    summary = summary.sort_values(
    by=['Votes', 'RF_Importance', 'Spearman', 'Kendall', 'DistCorr'], 
    ascending=False,
    kind='mergesort')
    summary = summary[~summary.duplicated(keep='first')]
    summary.to_csv(f'data/feature_summary_{n_top}_per_metric.csv', index=True)
    return summary

def plot_rank_and_aggregate_features_voting(n_top=10, summary=None):
    """ Plot the voting results for feature importance rankings. """
    if summary is None or summary.empty:
        raise ValueError("summary dataframe must be provided and non-empty")

    # Select top n_top features by votes
    top_features = summary.head(n_top)

    # Normalize values per column for better color scaling
    df_norm = top_features.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val != 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_norm, annot=top_features.round(3), cmap='YlGnBu', cbar=True, linewidths=0.5)
    plt.title(f"Top {n_top} Features based on Importance Voting Metrics")
    plt.ylabel("Features")
    plt.xlabel("Metrics")
    plt.savefig('figures/feature_importance_voting.pdf', bbox_inches='tight')
    plt.show()

