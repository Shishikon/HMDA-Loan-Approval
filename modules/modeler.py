from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from EDA import *






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




# action_map = {
#     'Loan_originated':           2,
#     'Approved_but_not_accepted': 1,
#     'Application_denied':        0
# }
# copy['action_taken'] = copy['action_taken'].map(action_map)


# X = copy.drop(columns=['action_taken'])
# y = copy['action_taken']
#
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
#
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_trainval, y_trainval, test_size=0.2,
#     random_state=RANDOM_STATE, stratify=y_trainval)




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
            zero_division=0)}



# ==============================Dummy START==============================
def baseline_model(X_train, y_train, X_valid, y_valid):
    dummy = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', DummyClassifier(strategy="most_frequent"))
    ])
    dummy.fit(X_train, y_train)
    baseline_valid = evaluator('Dummy Classifier',
                               dummy, X_valid, y_valid)
    return baseline_valid


# ==============================Dummy END==============================


# ==============================Logistic Regression START==============================

def log_model(X_train, y_train, X_valid, y_valid):
    logistic_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE))])
    logistic_pipeline.fit(X_train, y_train)
    logistic_valid = evaluator("Logistic model", logistic_pipeline, X_valid, y_valid)
    return logistic_valid


def log_balanced_model(X_train, y_train, X_valid, y_valid):
    logistic_pipeline_balanced = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE))])
    logistic_pipeline_balanced.fit(X_train, y_train)
    logistic_balanced_valid = evaluator("Logistic model Balanced", logistic_pipeline_balanced, X_valid, y_valid)
    return logistic_balanced_valid





# Confusion Matrix
def log_confusion_matrix(model, X_valid, y_valid):
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
def r_forest_unbalanced_scaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_unbalanced_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ))
    ])
    r_forest_unbalanced_pipeline.fit(X_train, y_train)
    r_forest_unbalanced_valid = evaluator("Non-balanced Random Forest",
                               r_forest_unbalanced_pipeline, X_valid, y_valid)
    return r_forest_unbalanced_valid


def r_forest_balanced_scaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_balanced_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])
    r_forest_balanced_pipeline.fit(X_train, y_train)
    r_forest_balanced_valid = evaluator("Balanced Random Forest",
                               r_forest_balanced_pipeline, X_valid, y_valid)
    return r_forest_balanced_valid


def r_forest_subsample_scaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_subsample_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE))])
    r_forest_subsample_pipeline.fit(X_train, y_train)
    r_forest_subsample_valid = evaluator("Subsample Random Forest",
                                         r_forest_subsample_pipeline, X_valid, y_valid)
    return r_forest_subsample_valid



def r_forest_unbalanced_unscaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_unbalanced_unscaled_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ))
    ])
    r_forest_unbalanced_unscaled_pipeline.fit(X_train, y_train)
    r_forest_unbalanced_unscaled_valid = evaluator("Non-balanced and Unscaled Random Forest",
                               r_forest_unbalanced_unscaled_pipeline, X_valid, y_valid)
    return r_forest_unbalanced_unscaled_valid


def r_forest_balanced_unscaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_balanced_scaled_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE))])
    r_forest_balanced_scaled_pipeline.fit(X_train, y_train)
    r_forest_balanced_unscaled_valid = evaluator("Balanced Unscaled Random Forest",
                               r_forest_balanced_scaled_pipeline, X_valid, y_valid)
    return r_forest_balanced_unscaled_valid


def r_forest_subsample_unscaled_model(X_train, y_train, X_valid, y_valid):
    r_forest_subsample_unscaled_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE))])
    r_forest_subsample_unscaled_pipeline.fit(X_train, y_train)
    r_forest_subsample_unscaled_valid = evaluator("Unscaled Subsample Random Forest",
                                         r_forest_subsample_unscaled_pipeline, X_valid, y_valid)
    return r_forest_subsample_unscaled_valid



















"""
def r_forest_confusion_matrix(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    cm = confusion_matrix(y_valid, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved Not Accepted', 'Originated'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Random Forest - Validation Set')
    plt.tight_layout()
    plt.show()

r_forest_confusion_matrix(model=r_forest_pipeline, X_valid=X_valid, y_valid=y_valid)



# ==============================Random Forest Classifier END==============================




# ==============================Decision Tree Classifier START==============================
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_not_scaled),
    ('model', DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=RANDOM_STATE))])

decision_tree_pipeline.fit(X_train, y_train)
decision_tree_valid = evaluator('Decision Tree', decision_tree_pipeline, X_valid, y_valid)



def decision_tree_confusion_matrix(model, X_valid, Y_valid):
    preds = model.predict(X_valid)
    cm = confusion_matrix(y_valid, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Denied', 'Approved Not Accepted', 'Originated'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Random Forest - Validation Set')
    plt.tight_layout()
    plt.show()



# ==============================Decision Tree Classifier START==============================


"""