"""Main script for fraud detection project using XGBoost and Optuna for hyperparameter tuning."""
from src.preprocessing import *
from src.model import *
from tests.reproduce import *
from src.visualisation import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from optuna.samplers import TPESampler
import optuna

if __name__ == "__main__":
    
    SEED = set_all_seeds(42) # set a fixed seed for reproducibility
    verify_reproducibility(quick_test=True, SEED=SEED)
    
    # load the data and process it 
    df = pd.read_csv('data/fraude_detection.csv')
    df_processed = preprocess_data(df) 
    feature_summary = rank_and_aggregate_features(df_processed, n_top=10, run_gridsearch=False, best_rf_params_path='data/best_rf_params.pkl')
    print(feature_summary.head(15))
    plot_rank_and_aggregate_features_voting(n_top=10, summary=feature_summary)

    # model building 
    feature_cols = feature_summary.index.tolist()[:10]  # use top 10 features for modeling
    X = df_processed[feature_cols]
    y = df_processed['target_enc']
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)
    
    # optuna study for XGBoost
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, X_val, X_test, y_train, y_val, y_test), n_trials=50)
    best_params = study.best_trial.params

    # train final XGBoost model based on the results of the optuna study and evaluate
    xgboost_model, y_test_pred, y_test_pred_proba = train_model(best_params, X_train, X_val, X_test, y_train, y_val)
    print("================================================================")
    print("XGBoost Model Evaluation:")
    evaluate_model(y_test, y_test_pred, y_test_pred_proba)
    
    # save XGBoost visualizations
    output_path = Path('figures')
    save_visualizations(xgboost_model, X_test, y_test, y_test_pred, y_test_pred_proba, output_path)

    # Logistic Regression benchmark + visualization
    print("================================================================")
    print("Logistic Regression Benchmark:")
    log_model, y_test_pred_log, y_test_pred_proba_log = logistic_regression_benchmark(X_train, X_test, y_train, y_test)
    plot_roc_auc_and_confusion(y_test, y_test_pred_proba_log, y_test_pred_log, model='Logistic Regression')

    