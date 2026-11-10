from catboost import CatBoostClassifier
from numpy.testing import print_coercion_tables
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from cleaner import one_value, cardinals, leaker, high_missing, transformer, filler_fit, filler_transform
from EDA import *
from modules.audit import auditor, summarizer
from mapper import mapper
from modules.modeler import (tune_log_reg, evaluator, tune_dec_tree,
                             tune_r_forest, tune_xgboost, plot_confusion_matrix, tune_catboost)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

raw = pd.read_csv('../actions_taken_1-2-3_loan_purposes_1_state_PA.csv', low_memory=False)

copy = raw.copy()



copy = one_value(copy)
copy = cardinals(copy)
copy = leaker(copy)
copy = high_missing(copy)
copy = transformer(copy)
copy = mapper(copy)


RANDOM_STATE = 42
action_map = {
    'Loan_originated':           2,
    'Approved_but_not_accepted': 1,
    'Application_denied':        0
}
copy['action_taken'] = copy['action_taken'].map(action_map)


X = copy.drop(columns=['action_taken'])
y = copy['action_taken']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_trainval)



from modeler import baseline_model
# _baseline = baseline_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)


from modeler import log_unbalanced_model, log_balanced_model

# _log_unbalanced = log_unbalanced_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
# _log_balanced = log_balanced_model(X_train=X_train, y_train=y_train,X_valid=X_valid, y_valid=y_valid)



from modeler import (r_forest_unbalanced_model, r_forest_balanced_model, r_forest_subsample_model)

# _r_forest_unbalanced_model = (
#     r_forest_unbalanced_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid))

# _r_forest_balanced_model = (
#     r_forest_balanced_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid))

# _r_forest_subsample_model = (
#     r_forest_subsample_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid))


from modeler import dec_tree_unbalanced_model, dec_tree_balanced_model

# _dec_tree_unbalanced_model = (
#     dec_tree_unbalanced_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid))

# _dec_tree_balanced_model = (
#     dec_tree_balanced_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid))



from modeler import xgboost_model
# _xgboost_model = xgboost_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)



# Tuning


# LogisticRegression

# best_log_model, best_log_params, log_reg_results = tune_log_reg(X_train, y_train)
# log_reg_summary = log_reg_results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
# log_validator = evaluator("Tuned log reg", best_log_model, X_valid, y_valid)
# conf_mat_log_reg = plot_confusion_matrix(best_log_model, X_valid, y_valid, 'Logistic Regression')

# DecisionTreeClassifier

# best_dtree_model, best_dtree_params, dtree_results = tune_dec_tree(X_train, y_train)
# dtree_summary = dtree_results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
# dtree_validator = evaluator("Tuned Decision Tree Classifier", best_dtree_model, X_valid, y_valid)
# conf_mat_dtree = plot_confusion_matrix(best_dtree_model, X_valid, y_valid, 'Decision Tree Classifier')

# RandomForestClassifier

best_r_forest_model, best_r_forest_params, r_forest_results = tune_r_forest(X_train, y_train)
r_forest_summary = r_forest_results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
r_forest_validator = evaluator("Tuned Random Forest Classifier", best_r_forest_model, X_valid, y_valid)
conf_mat_r_forest = plot_confusion_matrix(best_r_forest_model, X_valid, y_valid, 'Random Forest Classifier')
print(conf_mat_r_forest)
print(best_r_forest_params)

# XGBoostClassifier

# best_xgboost_model, best_xgboost_params, xgboost_results = tune_xgboost(X_train, y_train)
# xgboost_summary = xgboost_results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
# xgboost_validator = evaluator('Tuned XGBoost Classifier', best_xgboost_model, X_valid, y_valid)
# conf_mat_xgboost = plot_confusion_matrix(best_xgboost_model, X_valid, y_valid, 'XGBoost Classifier')
# print(best_xgboost_model)
# print(conf_mat_xgboost)

# CatBoostClassifier

best_catboost_model, best_catboost_params, catboost_results = tune_catboost(X_train, y_train)
catboost_summary = catboost_results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
catboost_validator = evaluator('Tuned CatBoost Classifier', best_catboost_model, X_valid, y_valid)
conf_mat_catboost = plot_confusion_matrix(best_catboost_model, X_valid, y_valid, 'CatBoost Classifier')
print(conf_mat_catboost)
print(best_catboost_params)
