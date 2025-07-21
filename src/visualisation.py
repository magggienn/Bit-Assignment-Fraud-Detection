""" This module contains functions for visualizing model performance and SHAP values.
It includes functions for plotting SHAP dependence plots, ROC curves, confusion matrices, and saving visualizations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_shap_dependence_grid(shap_values, X_test, feature_names, output_path):
    """
    Plot dependence plots for the six most influential features on a 2x3 grid.
    """
    # check if feature_names is provided, otherwise use the top features from shap_values
    if feature_names is None:
        top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-6:]
        feature_names = X_test.columns[top_indices]
    else:
        feature_names = feature_names[:6]

    # grid of SHAP dependence plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, feat in enumerate(feature_names):
        shap.dependence_plot(
            feat, shap_values, X_test, ax=axes[i], show=False
        )
        axes[i].set_title(feat)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_dependence_grid.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_roc_auc_and_confusion(y_test, y_test_pred_proba, y_test_pred, model='XGBoost'):
    """ 
    Plot ROC curve and confusion matrix (TP,TN,FP,FN) for the model predictions.
    """
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    # plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'{model} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    labels = [f'TN: {tn}', f'FP: {fp}', f'FN: {fn}', f'TP: {tp}']
    colors = ['#4CAF50', '#FF5722', '#FFC107', '#2196F3']

    ax = plt.gca()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.xticks([])
    plt.yticks([])

    ax.add_patch(plt.Rectangle((0, 1), 1, 1, color=colors[0], alpha=0.6))
    ax.text(0.5, 1.5, labels[0], ha='center', va='center', fontsize=14, fontweight='bold')
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, color=colors[1], alpha=0.6))
    ax.text(1.5, 1.5, labels[1], ha='center', va='center', fontsize=14, fontweight='bold')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=colors[2], alpha=0.6))
    ax.text(0.5, 0.5, labels[2], ha='center', va='center', fontsize=14, fontweight='bold')
    ax.add_patch(plt.Rectangle((1, 0), 1, 1, color=colors[3], alpha=0.6))
    ax.text(1.5, 0.5, labels[3], ha='center', va='center', fontsize=14, fontweight='bold')

    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'figures/roc_auc_and_confusion_{model}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def save_visualizations(final_model, X_test, y_test, y_test_pred, y_test_pred_proba, output_path):
    """
    Save SHAP visualizations including summary, dependence and waterfall plots for the top 6 features
    """
    true_positive_indices = [
        i for i in range(len(y_test))
        if y_test.iloc[i] == 1 and y_test_pred[i] == 1
    ]
    # true_positive_indices now contains all positions of correctly predicted frauds
    
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    
    # summary plot
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path / f'shap_summary.pdf', bbox_inches="tight")
    plt.close()
    
    # get the 6 most influential features by mean absolute SHAP value
    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-6:]
    top_feature_names = X_test.columns[top_indices]

    plot_shap_dependence_grid(shap_values, X_test, top_feature_names, output_path)
    plot_roc_auc_and_confusion(y_test, y_test_pred_proba, y_test_pred)
    
    # waterfall plot for the same sample  - true positive
    for i in range(5): #
        tp_idx = true_positive_indices[i]

        sample_case = X_test.iloc[tp_idx]
        print("Sample Fraud Case Features:\n", sample_case)
        print("True label:", y_test.iloc[tp_idx])
        print("Model prediction:", y_test_pred[tp_idx])
        print("Fraud probability:", y_test_pred_proba[tp_idx])
        sample_explanation = shap.Explanation(values=shap_values[tp_idx], 
                                            base_values=explainer.expected_value, 
                                            data=sample_case, 
                                            feature_names=X_test.columns)
        ax = shap.plots.waterfall(sample_explanation, max_display=20, show=False)
        fig = ax.get_figure()
        fig.savefig(output_path / f'shap_waterfall_sample_{tp_idx}.pdf', bbox_inches="tight")
        plt.close(fig)

def plot_rank_and_aggregate_features_voting(n_top=10, summary=None):
    """ 
    Plot the voting results for feature importance rankings. 
    """
    if summary is None or summary.empty:
        raise ValueError("summary dataframe must be provided and non-empty")

    # select top n_top features by votes
    top_features = summary.head(n_top)

    # normalize values per column for color scaling
    df_norm = top_features.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val != 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0

    # plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_norm, annot=top_features.round(3), cmap='YlGnBu', cbar=True, linewidths=0.5)
    plt.title(f"Top {n_top} Features based on Importance Voting Metrics")
    plt.ylabel("Features")
    plt.xlabel("Metrics")
    plt.savefig('figures/feature_importance_voting.pdf', bbox_inches='tight')
    plt.show()
