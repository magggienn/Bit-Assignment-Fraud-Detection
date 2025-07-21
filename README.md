# Bit Assignment - Fraud Detection
Hi! I'm Margarita, and this is my BIT assignment on Fraud Detection. Thank you for reviewing my work! 🙂

## File Structure
 The project's code is seperated in few folders that are somewhat self-explanatory - data (where the data files - given and saved are located), docs (where the documentation is), figures, notebooks, src (where the model, preprocessing and visualisation files are located), tests (reproducibility ensuring test), fraud_detection_env (this is the setup for the virtual environment!).
'''
fraud-detection/
├── data/                    # Dataset files (original and processed)
├── figures/                 # Generated plots and visualizations  
├── notebooks/               # Jupyter notebook for exploration
├── src/                     # Source code
│   ├── model.py            # Model training and evaluation
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   └── visualization.py    # Plotting functions
├── tests/                   # Testing file for reproducibility
├── fraud_detection_env/     # Virtual environment
├── main.py                 # Main execution script
└── requirements.txt        # Project dependencies
'''
## Setup
Prerequisites
Python 3.8+

To setup the virtual environment please follow these steps:
1. ### Activate the existing virtual environment
.\fraud_detection_env\Scripts\Activate.ps1

2. ### Install dependencies
pip install -r requirements.txt

3. ### Test your code runs
python main.py

4. ### Clean up environment
deactivate
Remove-Item -Recurse -Force fraud_detection_env

## Usage and Description
How to run the code? 
----> Everything is contained in the main.py file. Simply run this file! Be sure to follow the steps outlined in the previous section (SETUP).

The data is first preprocessed by imputing and cleaning. After this step, the rank_and_aggregate_features function is run to perform the correlation study. If you would like to run the Random Forest GridSearch optimization used in the correlation analysis, set the parameter run_gridsearch=True in the rank_and_aggregate_features function.

Based on the results of the voting correlation study, the top 10 most influential features are selected to build an XGBoost Classifier for predicting fraudulent behavior. To optimize the model, I used Optuna with 50 trials to find the best combination of parameters. The training data is split into training and validation sets. The final model was then trained and evaluated using the ROC-AUC metric, where XGBoost achieved a score of approximately 90%. A confusion matrix further illustrates the distribution of True Positives, False Negatives, True Negatives, and False Positives. For comparison, a basic Logistic Regression model was also built, which the XGBoost model outperformed.

To gain deeper insights into how individual features influence fraud prediction, I used SHAP for model explainability. I visualized a few True Positive cases to observe which features most strongly contributed to labeling the transactions as fraudulent. Additional plots show dependencies between important features.

Future Improvements:
- Add more tests to ensure model validity.
- Explore enhancements in precision, recall, and F1-score for class 1 ("fraud").
- Investigate models capable of handling multiclass classification, as there are three types of fraud.

To conclude - The XGBoost model demonstrated strong performance in detecting fraudulent behavior, achieving a ROC-AUC of around 90% and outperformed the Logistic Regression (86% ROC-AUC).