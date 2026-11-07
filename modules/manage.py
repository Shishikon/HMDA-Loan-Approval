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
_baseline = baseline_model(X_train=X_train, y_train=y_train,
                           X_valid=X_valid, y_valid=y_valid)



from modeler import log_model
_log = log_model(X_train=X_train, y_train=y_train,
                 X_valid=X_valid, y_valid=y_valid)


from modeler import log_balanced_model
_log_balanced = log_balanced_model(X_train=X_train, y_train=y_train,
                                   X_valid=X_valid, y_valid=y_valid)





from modeler import r_forest_unbalanced_scaled_model
_r_forest_unbalanced_scaled_model = r_forest_unbalanced_scaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)
print('Unbalanced-Scaled Random Forest', _r_forest_unbalanced_scaled_model)

from modeler import r_forest_balanced_scaled_model
_r_forest_balanced_scaled_model = r_forest_balanced_scaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)
print('Balanced-Scaled Random Forest', _r_forest_balanced_scaled_model)


from modeler import r_forest_subsample_scaled_model
_r_forest_subsample_scaled_model = r_forest_subsample_scaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)

print('Subsample-Scaled Random Forest', _r_forest_subsample_scaled_model)


from modeler import r_forest_unbalanced_unscaled_model
_r_forest_unbalanced_unscaled_model = r_forest_unbalanced_unscaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)
print('Unbalanced-Unscaled Random Forest', _r_forest_unbalanced_unscaled_model)


from modeler import r_forest_balanced_unscaled_model
_r_forest_balanced_unscaled_model = r_forest_balanced_unscaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)
print('Balanced-Unscaled Random Forest', _r_forest_balanced_unscaled_model)


from modeler import r_forest_subsample_unscaled_model
_r_forest_subsample_unscaled_model = r_forest_subsample_unscaled_model(X_train=X_train, y_train=y_train,
                                            X_valid=X_valid, y_valid=y_valid)

print('Subsample_balanced-Unscaled Random Forest', _r_forest_subsample_unscaled_model)


"""
Unbalanced-Scaled Random Forest {'model': 'Non-balanced Random Forest', 'accuracy': 0.8993628858578607, 'precision': 0.8845330399506218, 'recall': 0.8993628858578607, 'f1': 0.8808226454479658, 'macro_f1': 0.5725787883984179}
Balanced-Scaled Random Forest {'model': 'Balanced Random Forest', 'accuracy': 0.8920046661880833, 'precision': 0.8781077167413361, 'recall': 0.8920046661880833, 'f1': 0.88191682480703, 'macro_f1': 0.5955019571163371}
Subsample-Scaled Random Forest {'model': 'Subsample Random Forest', 'accuracy': 0.8974335965541995, 'precision': 0.8832062913586227, 'recall': 0.8974335965541995, 'f1': 0.8770937185986747, 'macro_f1': 0.5662586954827932}
Unbalanced-Unscaled Random Forest {'model': 'Non-balanced and Unscaled Random Forest', 'accuracy': 0.9002153625269204, 'precision': 0.8862592391093689, 'recall': 0.9002153625269204, 'f1': 0.8815770823404169, 'macro_f1': 0.5747637398511648}
Balanced-Unscaled Random Forest {'model': 'Balanced Unscaled Random Forest', 'accuracy': 0.8917354630294329, 'precision': 0.8782613295282796, 'recall': 0.8917354630294329, 'f1': 0.8819580893039005, 'macro_f1': 0.5980837256887269}
"""