import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

from cleaner import one_value, cardinals, leaker, high_missing, transformer, filler_fit, filler_transform
from EDA import *
from modules.audit import auditor, summarizer
from mapper import mapper
from modules.modeler import baseline_model

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
"""
{'model': 'Dummy Classifier', 
'accuracy': 0.860956568557071,
'precision': 0.7412462129415666,
'recall': 0.860956568557071,
'f1': 0.7966292448364942, 
'macro_f1': 0.3084279640925493}
"""

from modeler import log_model, log_balanced_model

# _log = log_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

# _log_balanced = log_balanced_model(X_train=X_train, y_train=y_train,X_valid=X_valid, y_valid=y_valid)



from modeler import (r_forest_unbalanced_scaled_model,
                     r_forest_balanced_scaled_model,
                     r_forest_subsample_scaled_model,
                     r_forest_unbalanced_unscaled_model,
                     r_forest_balanced_unscaled_model,
                     r_forest_subsample_unscaled_model)

# _r_forest_unbalanced_scaled_model = (
#     r_forest_unbalanced_scaled_model(X_train=X_train, y_train=y_train,
#                                      X_valid=X_valid, y_valid=y_valid))

# _r_forest_balanced_scaled_model = (
#     r_forest_balanced_scaled_model(X_train=X_train, y_train=y_train,
#                                    X_valid=X_valid, y_valid=y_valid))


# _r_forest_subsample_scaled_model = (
#     r_forest_subsample_scaled_model(X_train=X_train, y_train=y_train,
#                                     X_valid=X_valid, y_valid=y_valid))


# _r_forest_unbalanced_unscaled_model = (
#     r_forest_unbalanced_unscaled_model(X_train=X_train, y_train=y_train,
#                                        X_valid=X_valid, y_valid=y_valid))


# _r_forest_balanced_unscaled_model = (
#     r_forest_balanced_unscaled_model(X_train=X_train, y_train=y_train,
#                                      X_valid=X_valid, y_valid=y_valid))


# _r_forest_subsample_unscaled_model = (
#     r_forest_subsample_unscaled_model(X_train=X_train, y_train=y_train,
#                                       X_valid=X_valid, y_valid=y_valid))





from modeler import dec_tree_unbalanced_model, dec_tree_balanced_model


_dec_tree_unbalanced_model = dec_tree_unbalanced_model(X_train=X_train, y_train=y_train,
                                                       X_valid=X_valid, y_valid=y_valid)
print(_dec_tree_unbalanced_model)


_dec_tree_balanced_model = dec_tree_balanced_model(X_train=X_train, y_train=y_train,
                                                       X_valid=X_valid, y_valid=y_valid)
print(_dec_tree_balanced_model)

