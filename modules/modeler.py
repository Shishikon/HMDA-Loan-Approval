import numpy as np
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

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, classification_report)


# ============================== Modeling Global Config ==============================
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


# ============================== Preprocessing Pipelines ==============================
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
    ('cat', categorical_pipeline, categorical_cols)], remainder='drop')


# ============================== Central Evaluation Function ==============================
def evaluator(name, model, X_true, y_true):
    pred = model.predict(X_true)
    probabilities = model.predict_proba(X_true)

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

    sns.barplot(data=metrics_df, x='Score', y='Metric', hue='Metric', legend=False,
                ax=axes[0], palette='Blues_r')
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

    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('Class-wise Performance (Precision, Recall, F1)',
                      fontsize=12, fontweight='bold')
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


# ============================== Confusion Matrix ==============================
def plot_confusion_matrix(model, X_valid, y_valid, model_name):
    y_pred = model.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=True)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, pad=15)
    fig.tight_layout()
    plt.show()

    return cm


# ============================== Feature Importance ==============================
def plot_feature_importance(model, model_name, top_n=20):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['model']

    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        cat_feature_names = []

    feature_names = numeric_cols + cat_feature_names

    # Decision Tree, Random Forest, XGBoost
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        importance_label = 'Feature Importance (Gini / Gain)'

        # Logistic Regression - multiclass gives one row, takes the mean abs
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_).mean(axis=0)
        importance_label = 'Mean |Coefficient| Across Classes'
    else:
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axis('off')
        ax.text(0.5, 0.5, f'Feature importance not available for {type(classifier).__name__}',
                ha='center', va='center', fontsize=12, color='grey',
                transform=ax.transAxes)
        plt.tight_layout()
        plt.show()
        return None

    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature',
                hue='Feature', legend=False, palette='Blues_r', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(importance_label)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.show()
    return importance_df


# ============================== Threshold Analysis ==============================
def find_best_threshold(model, X_valid, y_valid, model_name, target_class):
    thresholds = np.arange(0.10, 0.90, 0.05)
    probabilities = model.predict_proba(X_valid)

    results = []
    for thresh in thresholds:
        # Override: if P(target_class) >= threshold, predict target_class
        preds = model.predict(X_valid).copy()
        preds[probabilities[:, target_class] >= thresh] = target_class

        f1_macro = f1_score(y_valid, preds, average='macro', zero_division=0)
        recall_target = recall_score(y_valid, preds, average=None,
                                     zero_division=0)[target_class]
        results.append({
            'threshold': round(thresh, 2),
            'macro_f1': f1_macro,
            f'recall_class_{target_class}': recall_target
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['macro_f1'].idxmax()]
    best_threshold = best_row['threshold']

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results_df['threshold'], results_df['macro_f1'],
            marker='o', color='steelblue', label='Macro F1')
    ax.plot(results_df['threshold'], results_df[f'recall_class_{target_class}'],
            marker='s', color='salmon', linestyle='--',
            label=f'Recall Class {target_class}')
    ax.axvline(best_threshold, color='green', linestyle=':', linewidth=2,
               label=f'Best threshold = {best_threshold:.2f}')
    ax.set_title(f'Threshold Analysis — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame([{
        'Model': model_name,
        'Target Class': target_class,
        'Best Threshold': f"{best_threshold:.2f}",
        'Macro F1 at Best Threshold': f"{best_row['macro_f1']:.4f}",
        f'Recall Class {target_class} at Best Threshold': f"{best_row[f'recall_class_{target_class}']:.4f}"}])

    fig, ax = plt.subplots(figsize=(10, 1.2))
    ax.axis('off')
    tbl = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c5f8a')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#d9eaf7')
    ax.set_title('Threshold Selection Summary', fontsize=11,
                 fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()

    return best_threshold, results_df


def predict_with_threshold(model, X, threshold, target_class):
    probabilities = model.predict_proba(X)
    preds = model.predict(X).copy()
    preds[probabilities[:, target_class] >= threshold] = target_class
    return preds


# ============================== Baseline ==============================
def baseline_model(X_train, y_train, X_valid, y_valid):
    dummy = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', DummyClassifier(strategy="most_frequent"))])

    dummy.fit(X_train, y_train)
    baseline = evaluator('Dummy Classifier', dummy, X_valid, y_valid)
    return baseline


# ============================== Tuning Functions ==============================
def tune_log_reg(X_train, y_train, X_valid, y_valid):
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_scaled),
        ('model', LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE))])

    params = {
        'model__C': [0.01, 1, 10, 100],
        'model__class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train, y_train)
    results_df = pd.DataFrame(grid_search.cv_results_)

    eval_results = evaluator('Logistic Regression (Tuned)',
                             grid_search.best_estimator_, X_valid, y_valid)

    plot_feature_importance(grid_search.best_estimator_,
                            'Logistic Regression (Tuned)')
    return grid_search.best_estimator_, grid_search.best_params_, results_df, eval_results


def tune_dec_tree(X_train, y_train, X_valid, y_valid):
    dec_tree_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE))])

    params = {
        'model__max_depth': [5, 10, 20],
        'model__class_weight': [None, 'balanced'],
        'model__min_samples_leaf': [1, 5],
        'model__criterion': ['gini', 'entropy']}

    grid_search = GridSearchCV(
        estimator=dec_tree_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train, y_train)
    dt_results_df = pd.DataFrame(grid_search.cv_results_)

    eval_results = evaluator('Decision Tree (Tuned)',
                             grid_search.best_estimator_, X_valid, y_valid)

    plot_feature_importance(grid_search.best_estimator_,
                            'Decision Tree (Tuned)')

    return grid_search.best_estimator_, grid_search.best_params_, dt_results_df, eval_results


def tune_r_forest(X_train, y_train, X_valid, y_valid):
    r_forest_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('model', RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1))])

    params = {
        'model__n_estimators': [200, 300],
        'model__max_depth': [10, 20, None],
        'model__class_weight': ['balanced', 'balanced_subsample'],
        'model__max_features': ['sqrt'],
        'model__min_samples_leaf': [1, 2]
    }
    # 2×3×2×1×2 = 24 combos × 3 CV = 72 fits

    grid_search = GridSearchCV(
        estimator=r_forest_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train, y_train)
    r_forest_results_df = pd.DataFrame(grid_search.cv_results_)

    eval_results = evaluator('Random Forest (Tuned)',
                             grid_search.best_estimator_, X_valid, y_valid)

    plot_feature_importance(grid_search.best_estimator_,
                            'Random Forest (Tuned)')

    return grid_search.best_estimator_, grid_search.best_params_, r_forest_results_df, eval_results


def tune_xgboost(X_train, y_train, X_valid, y_valid):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    xgboost_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor_not_scaled),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_delta_step=1))
    ])

    params = {
        'model__n_estimators': [300, 500],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.05, 0.1],
        'model__min_child_weight': [1, 5, 10],
        'model__subsample': [0.8]
    }
    # 2×3×2×3×1 = 36 combos × 3 CV = 108 fits
    # ~12–15 mins

    grid_search = GridSearchCV(
        estimator=xgboost_pipeline,
        param_grid=params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train, y_train)
    xgboost_results_df = pd.DataFrame(grid_search.cv_results_)

    eval_results = evaluator('XGBoost (Tuned)',
                             grid_search.best_estimator_, X_valid, y_valid)

    plot_feature_importance(grid_search.best_estimator_,
                            'XGBoost (Tuned)')

    return grid_search.best_estimator_, grid_search.best_params_, xgboost_results_df, eval_results


# ============================== Final Test Evaluation ==============================
def evaluate_winner_on_test(winner_model, winner_name, X_valid, y_valid,
                             X_test, y_test, target_class):

    # Find best threshold on validation
    best_threshold, threshold_df = find_best_threshold(
        winner_model, X_valid, y_valid, winner_name, target_class=target_class)

    # Apply threshold to test set - TEST SIZE
    y_pred_test = predict_with_threshold(
        winner_model, X_test, best_threshold, target_class=target_class)

    # Confusion matrix on test
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=True)
    ax.set_title(f'Test Set Confusion Matrix — {winner_name}', fontsize=14, pad=15)
    fig.tight_layout()
    plt.show()

    # Final metrics on test
    test_macro_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    test_bal_acc  = balanced_accuracy_score(y_test, y_pred_test)
    test_weighted_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_roc_auc  = roc_auc_score(
        y_test, winner_model.predict_proba(X_test), multi_class='ovr', average='macro')

    metrics_dict = {
        'Balanced Accuracy': test_bal_acc,
        'Weighted F1':       test_weighted_f1,
        'Macro F1':          test_macro_f1,
        'ROC-AUC (Macro)':   test_roc_auc}

    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Score'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.4]})
    fig.suptitle(f'Final Test Results — {winner_name}  |  Threshold {best_threshold:.2f} '
                 f'on class {target_class}', fontsize=14, fontweight='bold', y=1.01)

    sns.barplot(data=metrics_df, x='Score', y='Metric', hue='Metric',
                legend=False, ax=axes[0], palette='Blues_r')
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title('Headline Metrics (Test Set)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('')
    for p in axes[0].patches:
        w = p.get_width()
        axes[0].text(w + 0.02, p.get_y() + p.get_height() / 2,
                     f'{w:.4f}', va='center', ha='left', fontsize=10, fontweight='bold')

    report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).iloc[:-1, :-3].T
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=False,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('Per-Class Report (Test Set)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Classes')

    plt.tight_layout()
    plt.show()

    # Summary table
    summary_data = {
        'Model': [winner_name],
        'Set': ['Test'],
        'Threshold': [f"{best_threshold:.2f}"],
        'Macro F1': [f"{test_macro_f1:.4f}"],
        'Balanced Acc': [f"{test_bal_acc:.4f}"],
        'Weighted F1': [f"{test_weighted_f1:.4f}"],
        'ROC-AUC': [f"{test_roc_auc:.4f}"]
    }
    summary_df = pd.DataFrame(summary_data)

    fig2, ax2 = plt.subplots(figsize=(12, 1.2))
    ax2.axis('off')
    tbl = ax2.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1a3f5c')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#cce0f0')
    ax2.set_title('Final Test Score Summary', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()

    return {
        'best_threshold': best_threshold,
        'threshold_analysis': threshold_df,
        'confusion_matrix': cm,
        'macro_f1': test_macro_f1,
        'roc_auc': test_roc_auc}

