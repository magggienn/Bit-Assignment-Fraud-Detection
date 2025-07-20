from preprocessing import *

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Load your DataFrame here
    df = pd.read_csv('data/fraude_detection.csv')
    df_processed = preprocess_data(df)  # preprocess your DataFrame first
    feature_summary = rank_and_aggregate_features(df_processed, n_top=10)
    print(feature_summary.head(15))
    plot_rank_and_aggregate_features_voting(n_top=10, summary=feature_summary)
