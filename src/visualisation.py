import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_shap_dependence_grid(shap_values, X_test, feature_names, output_path):
    """
    Plot dependence plots for the six most influential features on a 2x3 grid.
    """
    # Use the current order if you've already sorted your top features,
    # otherwise, sort by mean absolute SHAP value:
    if feature_names is None:
        top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-6:]
        feature_names = X_test.columns[top_indices]
    else:
        feature_names = feature_names[:6]

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
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'{model} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Confusion Matrix
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
    """Save SHAP visualizations including summary, bar, and dependence plots for top features"""
    
    true_positive_indices = [
        i for i in range(len(y_test))
        if y_test.iloc[i] == 1 and y_test_pred[i] == 1
    ]
    # true_positive_indices now contains all positions of correctly predicted frauds
    
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path / f'shap_summary.pdf', bbox_inches="tight")
    plt.close()
    
    # Get the 6 most influential features by mean absolute SHAP value
    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-6:]
    top_feature_names = X_test.columns[top_indices]

    plot_shap_dependence_grid(shap_values, X_test, top_feature_names, output_path)
    plot_roc_auc_and_confusion(y_test, y_test_pred_proba, y_test_pred)
    
    # Waterfall plot for the same sample  - true positive
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
