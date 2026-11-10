import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, classification_report)
from catboost import CatBoostClassifier



# Modeling Global Config
RANDOM_STATE = 42

numeric_cols = [
    'loan_amount', 'loan_to_value_ratio', 'loan_term', 'property_value', 
    'income', 'debt_to_income_ratio', 'tract_population', 
    'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 
    'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 
    'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units', 
    'applicant_age_not_applicable', 'applicant_age_ord', 
    'coapplicant_age_not_applicable', 'coapplicant_age_ord', 
    'is_single_family', 'no_coapplicant'
]

categorical_cols = [
    'conforming_loan_limit', 'derived_loan_product_type', 'derived_ethnicity',
    'derived_race', 'derived_sex', 'preapproval', 'loan_type', 'lien_status',
    'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose',
    'negative_amortization', 'interest_only_payment', 'balloon_payment', 
    'other_nonamortizing_features', 'construction_method', 'occupancy_type', 
    'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 
    'applicant_credit_score_type', 'co-applicant_credit_score_type', 
    'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed',
    'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1',
    'applicant_race_observed', 'co-applicant_race_observed', 'applicant_sex',
    'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed',
    'applicant_age_above_62', 'submission_of_application',
    'initially_payable_to_institution', 'aus-1', 'county_code',
    'derived_dwelling_category', 'derived_msa-md'
]

# Pipelines for Data Preprocessing
numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

numeric_not_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor_scaled = ColumnTransformer(transformers=[
    ('num', numeric_scaled, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)], remainder='drop')

preprocessor_not_scaled = ColumnTransformer(transformers=[
    ('num', numeric_not_scaled, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),], remainder='drop')


# Central Evaluation Function
def evaluator(name, model, X_true, y_true):
    pred = model.predict(X_true)
    probabilities = model.predict_proba(X_true)

    # Calculate metrics
    acc = accuracy_score(y_true, pred)
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro')

    report_str = classification_report(y_true, pred, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle(f'Model Evaluation Report: {name}', fontsize=16, fontweight='bold', y=0.98)

    metrics_dict = {
        'Accuracy': acc,
        'Balanced Acc': bal_acc,
        'Weighted F1': f1,
        'Macro F1': macro_f1,
        'ROC-AUC (Macro)': roc_auc
    }
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Score'])

    sns.barplot(data=metrics_df, x='Score', y='Metric', hue='Metric', legend=False, ax=axes[0], palette='Blues_r')
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title('Core Performance Metrics', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Score Value')
    axes[0].set_ylabel('')

    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(width + 0.02, p.get_y() + p.get_height() / 2, f'{width:.3f}',
                     va='center', ha='left', fontsize=10, fontweight='bold')

    report_dict = classification_report(y_true, pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).iloc[:-1, :-3].T

    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False, ax=axes[1], linewidths=0.5)
    axes[1].set_title('Class-wise Performance (Precision, Recall, F1)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Classes')

    plt.tight_layout()
    plt.show()

    return {
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "classification_report": report_str}


# ============================== Dummy Classifier ==============================

def baseline_model(X_train, y_train, X_valid, y_valid):
    dummy = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', DummyClassifier(strategy="most_frequent"))])

    dummy.fit(X_train, y_train)
    baseline = evaluator('Dummy Classifier', dummy, X_valid, y_valid)
    return baseline

# ============================== Logistic Regression ==============================
def log_unbalanced_model(X_train, y_train, X_valid, y_valid):
    logistic_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE))])

    logistic_pipeline.fit(X_train, y_train)

    log_unbalanced_valid = evaluator("Unbalanced Logistic Regression", logistic_pipeline, X_valid, y_valid)
    return log_unbalanced_valid


def log_balanced_model(X_train, y_train, X_valid, y_valid):
    logistic_pipeline_balanced = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE))])

    logistic_pipeline_balanced.fit(X_train, y_train)
    log_balanced_valid = evaluator("Balanced Logistic Regression", logistic_pipeline_balanced, X_valid, y_valid)
    return log_balanced_valid


# ============================== Decision Tree Classifier ==============================
def dec_tree_unbalanced_model(X_train, y_train, X_valid, y_valid):
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE))])

    dt_pipeline.fit(X_train, y_train)
    dec_tree_unbalanced_valid =  evaluator('Unbalanced Decision Tree', dt_pipeline, X_valid, y_valid)
    return dec_tree_unbalanced_valid


def dec_tree_balanced_model(X_train, y_train, X_valid, y_valid):
    dt_balanced_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE))])

    dt_balanced_pipeline.fit(X_train, y_train)
    dec_tree_balanced_valid =  evaluator('Balanced Decision Tree', dt_balanced_pipeline, X_valid, y_valid)
    return dec_tree_balanced_valid


# ============================== Random Forest Classifier ==============================
def r_forest_unbalanced_model(X_train, y_train, X_valid, y_valid):
    r_forest_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=RANDOM_STATE))])

    r_forest_pipeline.fit(X_train, y_train)
    r_forest_unbalanced_valid =  evaluator("Unbalanced-Unscaled Random Forest",
                                           r_forest_pipeline, X_valid, y_valid)
    return r_forest_unbalanced_valid


def r_forest_balanced_model(X_train, y_train, X_valid, y_valid):
    r_forest_balanced_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE))])

    r_forest_balanced_pipeline.fit(X_train, y_train)
    r_forest_balanced_valid = evaluator("Balanced-Unscaled Random Forest",
                                        r_forest_balanced_pipeline, X_valid, y_valid)
    return r_forest_balanced_valid


def r_forest_subsample_model(X_train, y_train, X_valid, y_valid):
    r_forest_subsample_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=RANDOM_STATE))])

    r_forest_subsample_pipeline.fit(X_train, y_train)
    r_forest_subsample_valid =  evaluator("Subsample_balanced-Unscaled Random Forest",
                                          r_forest_subsample_pipeline, X_valid, y_valid)
    return r_forest_subsample_valid


# ============================== XGBoost Classifier ==============================
def xgboost_model(X_train, y_train, X_valid, y_valid):
    xgboost_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1))])

    xgboost_pipeline.fit(X_train, y_train)
    xgboost_valid =  evaluator('XGBoost', xgboost_pipeline, X_valid, y_valid)
    return xgboost_valid



# Tuning

def tune_log_reg(X_train, y_train):
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs'))])

    params = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__class_weight': [None, 'balanced'],
        'model__solver': ['lbfgs'],
        'model__max_iter': [2000]
    }

    best_params_conf_matrix = {
        'model__C': [0.01],
        'model__class_weight': ['balanced']}

    grid_search = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)
    grid_search.fit(X_train, y_train)
    results_df = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, grid_search.best_params_, results_df


def tune_dec_tree(X_train, y_train):
    dec_tree_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', DecisionTreeClassifier(
            random_state=RANDOM_STATE))])

    params = {
        'model__max_depth': [5, 10, None],
        'model__class_weight': ['balanced'],
        'model__max_features': [None, 'sqrt', 0.5],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 3, 5],
        'model__criterion': ['gini', 'entropy']
    }

    best_params_conf_matrix = {
        'model__max_depth': [10],
        'model__class_weight': ['balanced'],
        'model__max_features': [0.5],
        'model__min_samples_split': [5],
        'model__min_samples_leaf': [3],
        'model__criterion': ['gini']
    }

    grid_search = GridSearchCV(
        estimator=dec_tree_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train, y_train)
    dt_results_df = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, grid_search.best_params_, dt_results_df


def tune_r_forest(X_train, y_train):
    # from imblearn.over_sampling import SMOTE
    # from imblearn.pipeline import Pipeline

    r_forest_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        # ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1))])

    params = {
        'model__n_estimators': [200, 500],
        'model__max_depth': [10, 20, None],
        'model__class_weight': ['balanced', 'balanced_subsample'],
        'model__max_features': ['sqrt', 0.5],
        'model__min_samples_leaf': [1, 4],
        'model__min_samples_split': [2, 10]
    }

    best_params_conf_matrix = {
        'model__n_estimators': [300],
        'model__max_depth': [10, 15],
        'model__class_weight': ['balanced'],
        'model__max_features': ['sqrt', 0.5],
        'model__min_samples_leaf': [1, 3]
    }

    grid_search = GridSearchCV(
        estimator=r_forest_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)
    grid_search.fit(X_train, y_train)
    r_forest_results_df = pd.DataFrame(grid_search.cv_results_)
    return grid_search.best_estimator_, grid_search.best_params_, r_forest_results_df


def tune_xgboost(X_train, y_train):
    xgboost_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1))])

    params = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }
    best_params_conf_matrix = {
        'model__n_estimators': [300],
        'model__max_depth': [5],
        'model__learning_rate': [0.01],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgboost_pipeline,
        param_grid=best_params_conf_matrix,
        cv=3,
        scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
        refit = 'f1_macro',
        n_jobs=-1,
        verbose=1)
    grid_search.fit(X_train, y_train)
    xgboost_results_df = pd.DataFrame(grid_search.cv_results_)
    return grid_search.best_estimator_, grid_search.best_params_, xgboost_results_df



def tune_catboost(X_train, y_train):
    catboost_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', CatBoostClassifier(
            random_seed=RANDOM_STATE,
            thread_count=-1,
            # Set to 1 so GridSearchCV's n_jobs=-1 handles parallelization cleanly without thread conflicts
            verbose=False,  # Keeps training logs clean during grid search
         bootstrap_type='Bernoulli',
            auto_class_weights='Balanced'
        ))
    ])

    # CatBoost parameter equivalents
    params = {
        'model__iterations': [300],  # Equivalent to n_estimators
        'model__depth': [3, 5],  # Equivalent to max_depth
        'model__learning_rate': [0.01],
        'model__subsample': [0.8]
    }

    grid_search = GridSearchCV(
        estimator=catboost_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    catboost_results_df = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, grid_search.best_params_, catboost_results_df


# Confusion Matrix
def plot_confusion_matrix(model, X_valid, y_valid, model_name):
    y_pred = model.predict(X_valid)
    cm = confusion_matrix(
        y_valid,
        y_pred,
        labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1, 2])

    disp.plot(
        cmap='Blues',
        values_format='d',
        ax=ax,
        colorbar=True
    )

    ax.set_title(f'Confusion Matrix - {model_name}',
        fontsize=14, pad=15)
    fig.tight_layout()
    plt.show()

    return cm














