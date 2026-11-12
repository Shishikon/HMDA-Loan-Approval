import pandas as pd
from sklearn.model_selection import train_test_split

from cleaner import one_value, cardinals, leaker, high_missing, transformer
from EDA import *
from modules.audit import auditor, summarizer
from mapper import mapper
from modeler import (
    baseline_model,
    tune_log_reg,
    tune_dec_tree,
    tune_r_forest,
    tune_xgboost,
    evaluate_winner_on_test
)
from modules.modeler import plot_confusion_matrix

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ============================== Data Loading ==============================

raw = pd.read_csv('../actions_taken_1-2-3_loan_purposes_1_state_PA.csv', low_memory=False)
copy = raw.copy()


# ============================== Cleaning & Preprocessing ==============================

copy = one_value(copy)
copy = cardinals(copy)
copy = leaker(copy)
copy = high_missing(copy)
copy = transformer(copy)
copy = mapper(copy)


# ============================== Target Encoding ==============================

RANDOM_STATE = 42

action_map = {
    'Loan_originated':           2,
    'Approved_but_not_accepted': 1,
    'Application_denied':        0
}
copy['action_taken'] = copy['action_taken'].map(action_map)


# ============================== Train / Validation / Test Split ==============================

X = copy.drop(columns=['action_taken'])
y = copy['action_taken']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_trainval)


# ============================== Baseline ==============================

# baseline_model(X_train, y_train, X_valid, y_valid)


# ============================== Tuning — All Four Models ==============================

# log_best_model, log_best_params, log_cv, log_eval = tune_log_reg(X_train, y_train, X_valid, y_valid)
# d_tree_best_model, d_tree_best_params, d_tree_cv, d_tree_eval = tune_dec_tree(X_train, y_train, X_valid, y_valid)
# r_forest_best_model, r_forest_best_params, r_forest_cv, r_forest_eval = tune_r_forest(X_train, y_train, X_valid, y_valid)
# xgb_best_model, xgb_best_params, xgb_cv, xgb_eval = tune_xgboost(X_train, y_train, X_valid, y_valid)

# ============================== Validation Comparison ==============================

# comparison = pd.DataFrame([log_eval, d_tree_eval, r_forest_eval, xgb_eval])
# comparison[['model', 'macro_f1', 'roc_auc', 'balanced_accuracy']].sort_values(
#     by='macro_f1', ascending=False)
# print(comparison.to_string())

# ============================== Final Test Evaluation ==============================

# test_results = evaluate_winner_on_test(
#     winner_model = None,
#     winner_name  = 'XGBoost',
#     X_valid      = X_valid,
#     y_valid      = y_valid,
#     X_test       = X_test,
#     y_test       = y_test,
#     target_class = 0)               # 0=Denied, 1=Approved-not-accepted, 2=Originated
# print(test_results)


# plot_confusion_matrix(log_best_model, X_valid, y_valid, 'Logistic Regression (Tuned)')
# plot_confusion_matrix(d_tree_best_model, X_valid, y_valid, 'Decision Tree (Tuned)')
# plot_confusion_matrix(r_forest_best_model, X_valid, y_valid, 'Random Forest (Tuned)')
# plot_confusion_matrix(xgb_best_model, X_valid, y_valid, 'XGBoost (Tuned)')











