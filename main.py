from src.preprocessing import *
from src.model import *
from tests.reproduce import *
from src.visualisation import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from optuna.samplers import TPESampler

if __name__ == "__main__":
    
    SEED = set_all_seeds(42) # Set a fixed seed for reproducibility
    verify_reproducibility(quick_test=False, SEED=SEED)
    
    # DataFrame 
    df = pd.read_csv('data/fraude_detection.csv')
    df_processed = preprocess_data(df)  # preprocess your DataFrame first
    feature_summary = rank_and_aggregate_features(df_processed, n_top=10, run_gridsearch=False, best_rf_params_path='data/best_rf_params.pkl')
    print(feature_summary.head(15))
    plot_rank_and_aggregate_features_voting(n_top=10, summary=feature_summary)

    # Model building 
    feature_cols = feature_summary.index.tolist()[:10]  # Use top 10 features for modeling
    X = df_processed[feature_cols]
    y = df_processed['target_enc']
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)
    
    # Optuna study for XGBoost
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, X_val, X_test, y_train, y_val, y_test), n_trials=50)
    best_params = study.best_trial.params

    # Train final XGBoost model and evaluate
    xgboost_model, y_test_pred, y_test_pred_proba = train_model(best_params, X_train, X_val, X_test, y_train, y_val)
    print("================================================================")
    print("XGBoost Model Evaluation:")
    evaluate_model(y_test, y_test_pred, y_test_pred_proba)
    
    # Save visualizations
    output_path = Path('figures')
    save_visualizations(xgboost_model, X_test, y_test, y_test_pred, y_test_pred_proba, output_path)

    # Logistic Regression benchmark
    print("================================================================")
    print("Logistic Regression Benchmark:")
    log_model, y_test_pred_log, y_test_pred_proba_log = logistic_regression_benchmark(X_train, X_test, y_train, y_test)
    plot_roc_auc_and_confusion(y_test, y_test_pred_proba_log, y_test_pred_log, model='Logistic Regression')

    