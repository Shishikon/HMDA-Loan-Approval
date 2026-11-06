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
from cleaner import one_value, cardinals, leaker, high_missing, transformer, filler_fit, filler_transform
from EDA import *
from modules.audit import auditor, summarizer
from mapper import mapper




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






# Modeling
RANDOM_STATE = 42

numeric_cols = ['loan_amount', 'loan_to_value_ratio',
    'loan_term', 'property_value', 'income', 'debt_to_income_ratio',
    'tract_population', 'tract_minority_population_percent',
    'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage',
    'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units', 'applicant_age_not_applicable',
    'applicant_age_ord', 'coapplicant_age_not_applicable', 'coapplicant_age_ord',
    'is_single_family', 'no_coapplicant'
]


categorical_cols = ['conforming_loan_limit', 'derived_loan_product_type', 'derived_ethnicity',
    'derived_race', 'derived_sex', 'preapproval', 'loan_type', 'lien_status',
    'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose',
    'negative_amortization', 'interest_only_payment',
    'balloon_payment', 'other_nonamortizing_features', 'construction_method',
    'occupancy_type', 'manufactured_home_secured_property_type',
    'manufactured_home_land_property_interest', 'applicant_credit_score_type',
    'co-applicant_credit_score_type', 'applicant_ethnicity-1',
    'co-applicant_ethnicity-1', 'applicant_ethnicity_observed',
    'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1',
    'applicant_race_observed', 'co-applicant_race_observed', 'applicant_sex',
    'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
    'applicant_age_above_62', 'submission_of_application',
    'initially_payable_to_institution', 'aus-1', 'county_code',
    'derived_dwelling_category', 'derived_msa-md'
]




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




numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
numeric_not_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor_scaled = ColumnTransformer(transformers=[
    ('num', numeric_scaled, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
], remainder='drop')
preprocessor_not_scaled = ColumnTransformer(transformers=[
    ('num', numeric_not_scaled, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
], remainder='drop')


def evaluator(name, model, X_data, y_true):
    pred = model.predict(X_data)

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred,
            average='weighted',
            zero_division=0),
        "recall": recall_score(y_true, pred,
            average='weighted',
            zero_division=0),
        "f1": f1_score(y_true, pred,
                       average='weighted',
                       zero_division=0),
        "macro_f1": f1_score(y_true, pred,
            average='macro',
            zero_division=0
        )
    }



# ==============================Dummy START==============================
dummy = Pipeline(steps=[
    ('preprocessor', preprocessor_scaled),
    ('model', DummyClassifier(strategy="most_frequent"))
])
dummy.fit(X_train, y_train)
baseline_valid = evaluator("Dummy baseline", dummy, X_valid, y_valid)
"""
{'model': 'Dummy baseline',
'accuracy': 0.860956568557071,
'precision': 0.7412462129415666,
'recall': 0.860956568557071,
'f1': 0.7966292448364942},
'macro_f1': 0.3084279640925493
"""
# ==============================Dummy END==============================



# ==============================Logistic Regression START==============================
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_scaled),
    ('model', LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE))])
logistic_pipeline_balanced = Pipeline(steps=[
    ('preprocessor', preprocessor_scaled),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE

    ))
])

logistic_pipeline.fit(X_train, y_train)
logistic_pipeline_balanced.fit(X_train, y_train)

logistic_valid = evaluator("Logistic model", logistic_pipeline, X_valid, y_valid)
"""
Before leakers dropped:

{'model': 'Logistic model',
'accuracy': 0.9551776740847092,
'precision': 0.9544206818326683,
'recall': 0.9551776740847092,
'f1': 0.9497421884803898,
'macro_f1': 0.7341193255858962}

After leakers dropped:

{'model': 'Logistic model', 
'accuracy': 0.8898061737257718, 
'precision': 0.869006518127738, 
'recall': 0.8898061737257718, 
'f1': 0.8640432138868192, 
'macro_f1': 0.4932932257230885}
"""

"""
Before leakers dropped:

{'model': 'Logistic model Balanced',
'accuracy': 0.8979720028715004,
'precision': 0.9428364493038996,
'recall': 0.8979720028715004,
'f1': 0.91631674805584,
'macro_f1': 0.6650087487152049}


After leakers dropped:

{'model': 'Logistic model Balanced', 
'accuracy': 0.6996141421392678, 
'precision': 0.8391852979152764, 
'recall': 0.6996141421392678, 
'f1': 0.7535906109649437, 
'macro_f1': 0.4418807012886437}
"""
logistic_balanced_valid = evaluator("Logistic model Balanced", logistic_pipeline_balanced, X_valid, y_valid)

# Confusion Matrix
def log_confusion_matrix(model, X_valid, y_valid, logger=None):
    preds = model.predict(X_valid)
    cm = confusion_matrix(y_valid, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved Not Accepted', 'Originated'])
    disp.plot(cmap='Blues')
    plt.title('Logistic Regression - Validation Set')
    plt.tight_layout()
    plt.show()

# log_confusion_matrix(model=logistic_pipeline, X_valid=X_valid, y_valid=y_valid, logger=logistic_pipeline)
# log_confusion_matrix(model=logistic_pipeline, X_valid=X_valid, y_valid=y_valid, logger=logistic_pipeline_balanced)

# ==============================Logistic Regression END==============================



# ==============================Random Forest Classifier START==============================


# Random Forest Classifier
r_forest = Pipeline(steps=[
    ('preprocessor', preprocessor_not_scaled),
    ('model', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE))])

r_forest.fit(X_train, y_train)
r_forest_valid = evaluator("Random Forest", r_forest, X_valid, y_valid)

"""
Before leakers droped:

{'model': 'Random Forest', 
'accuracy': 0.9899497487437185, 
'precision': 0.9898800527729477, 
'recall': 0.9899497487437185, 
'f1': 0.9895653430341331, 
'macro_f1': 0.9447280301513942}

After leakers droped:
{'model': 'Random Forest', 
'accuracy': 0.9002153625269204, 
'precision': 0.8862592391093689, 
'recall': 0.9002153625269204, 
'f1': 0.8815770823404169, 
'macro_f1': 0.5747637398511648}

After class_weight is balanced:
{'model': 'Random Forest', 
'accuracy': 0.8917354630294329, 
'precision': 0.8782613295282796, 
'recall': 0.8917354630294329, 
'f1': 0.8819580893039005, 
'macro_f1': 0.5980837256887269}

"""

def r_random_confusion_matrix(model, X_valid, y_valid, logger=None):
    preds = model.predict(X_valid)
    cm = confusion_matrix(y_valid, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved Not Accepted', 'Originated'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Random Forest - Validation Set')
    plt.tight_layout()
    plt.show()

# log_confusion_matrix(model=logistic_pipeline, X_valid=X_valid, y_valid=y_valid, logger=logistic_pipeline)


print(r_forest_valid)
# ==============================Random Forest Classifier END==============================



