import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



TARGET = 'action_taken'
ORDER  = ['Loan_originated', 'Approved_but_not_accepted', 'Application_denied']
COLORS = ['#2ecc71', '#f39c12', '#e74c3c']


def plot_target_distribution(df):
    counts = df[TARGET].value_counts()
    values = [counts[label] for label in ORDER]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ORDER, values, color=COLORS)

    for bar, val in zip(bars, values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    ax.set_title('Action Taken - Class Distribution')
    ax.set_ylabel('Count')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Originated', 'Approved Not Accepted', 'Denied'])
    plt.tight_layout()
    plt.show()


def plot_boxplots_numeric(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Income — clip between 0 and 500 (realistic income range in $k)
    ax = axes[0][0]
    plot_df = df[(df['income'] >= 0) & (df['income'] <= 500)]
    sns.boxplot(data=plot_df, x=TARGET, y='income',
                order=ORDER, hue=TARGET, palette=COLORS,
                legend=False, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title('Income (clipped 0–500k)')
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Originated', 'Approved\nNot Accepted', 'Denied'], fontsize=8)

    ax = axes[0][1]
    plot_df = df[df['loan_amount'] <= 1_000_000]
    sns.boxplot(data=plot_df, x=TARGET, y='loan_amount',
                order=ORDER, hue=TARGET, palette=COLORS,
                legend=False, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title('Loan Amount (clipped at $1M)')
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Originated', 'Approved\nNot Accepted', 'Denied'], fontsize=8)

    ax = axes[1][0]
    sns.boxplot(data=df, x=TARGET, y='interest_rate',
                order=ORDER, hue=TARGET, palette=COLORS,
                legend=False, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title('Interest Rate')
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Originated', 'Approved\nNot Accepted', 'Denied'], fontsize=8)

    ax = axes[1][1]
    dti_counts = df.groupby([TARGET, 'debt_to_income_ratio']).size().reset_index(name='count')
    for outcome, color in zip(ORDER, COLORS):
        subset = dti_counts[dti_counts[TARGET] == outcome]
        ax.bar(subset['debt_to_income_ratio'], subset['count'],
               alpha=0.6, color=color, label=outcome.replace('_', ' '), width=0.4)
    ax.set_title('Debt-to-Income Ratio Distribution')
    ax.set_xlabel('DTI Value')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_ltv_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4))

    # clip to realistic LTV range, anything above 105 is dirty data
    for outcome, color in zip(ORDER, COLORS):
        values = df[df[TARGET] == outcome]['loan_to_value_ratio'].dropna()
        values = values[values <= 105]
        ax.hist(values, bins=60, alpha=0.5, color=color,
                label=outcome.replace('_', ' '), density=True)

    ax.axvline(80, color='blue', linestyle='--', linewidth=1, label='LTV = 80%')
    ax.axvline(100, color='black', linestyle='--', linewidth=1, label='LTV = 100%')
    ax.set_title('Loan-to-Value Ratio by Outcome')
    ax.set_xlabel('LTV Ratio (%)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter_income_loan(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = df[(df['income'] >= 0) & (df['income'] <= 500)]
    plot_df = plot_df[plot_df['loan_amount'] <= 1_000_000]

    for outcome, color in zip(ORDER, COLORS):
        subset = plot_df[plot_df[TARGET] == outcome].sample(min(2000, len(plot_df)), random_state=42)
        ax.scatter(subset['income'], subset['loan_amount'],
                   alpha=0.25, s=10, color=color,
                   label=outcome.replace('_', ' '))

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 600_000)
    ax.set_title('Income vs Loan Amount by Outcome')
    ax.set_xlabel('Income ($k)')
    ax.set_ylabel('Loan Amount ($)')
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.show()


def plot_loantype_vs_outcome(df):
    rate_df = df.groupby(['loan_type', TARGET]).size().reset_index(name='count')
    rate_df['rate'] = rate_df['count'] / rate_df.groupby('loan_type')['count'].transform('sum') * 100

    loan_types = df['loan_type'].value_counts().index.tolist()
    x = np.arange(len(loan_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (outcome, color) in enumerate(zip(ORDER, COLORS)):
        vals = []
        for lt in loan_types:
            row = rate_df[(rate_df['loan_type'] == lt) & (rate_df[TARGET] == outcome)]['rate']
            vals.append(row.values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=outcome.replace('_', ' '), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(loan_types, rotation=15)
    ax.set_title('Loan Type vs Outcome Rate (%)')
    ax.set_ylabel('Percentage (%)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_denial_by_race(df):
    race_df = df.groupby(['derived_race', TARGET]).size().reset_index(name='count')
    race_df['rate'] = race_df['count'] / race_df.groupby('derived_race')['count'].transform('sum') * 100

    total_per_race = df['derived_race'].value_counts()
    valid_races = total_per_race[total_per_race >= 100].index
    race_df = race_df[race_df['derived_race'].isin(valid_races)]

    denial_rates = (race_df[race_df[TARGET] == 'Application_denied']
                    .set_index('derived_race')['rate']
                    .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(denial_rates.index, denial_rates.values, color='#e74c3c', edgecolor='white')

    for bar, val in zip(bars, denial_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_title('Denial Rate by Race (groups with 100+ applications)')
    ax.set_ylabel('Denial Rate (%)')
    ax.set_xticks(range(len(denial_rates)))
    ax.set_xticklabels(denial_rates.index, rotation=25, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_preapproval_vs_outcome(df):
    pre_df = df.groupby(['preapproval', TARGET]).size().reset_index(name='count')
    pre_df['rate'] = pre_df['count'] / pre_df.groupby('preapproval')['count'].transform('sum') * 100

    preapprovals = df['preapproval'].value_counts().index.tolist()
    x = np.arange(len(preapprovals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (outcome, color) in enumerate(zip(ORDER, COLORS)):
        vals = []
        for p in preapprovals:
            row = pre_df[(pre_df['preapproval'] == p) & (pre_df[TARGET] == outcome)]['rate']
            vals.append(row.values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=outcome.replace('_', ' '), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(preapprovals)
    ax.set_title('Preapproval Status vs Outcome Rate (%)')
    ax.set_ylabel('Percentage (%)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_neighborhood_income(df):
    plot_df = df[df['ffiec_msa_md_median_family_income'] > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x=TARGET, y='ffiec_msa_md_median_family_income',
                order=ORDER, hue=TARGET, palette=COLORS,
                legend=False, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))

    ax.set_title('Neighborhood Median Family Income vs Outcome')
    ax.set_xlabel('')
    ax.set_ylabel('Median Family Income ($)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Originated', 'Approved Not Accepted', 'Denied'])
    plt.tight_layout()
    plt.show()


def plot_minority_tract_denial(df):
    temp = df.copy()
    temp['minority_bin'] = pd.cut(temp['tract_minority_population_percent'],
                                   bins=[0, 10, 25, 50, 75, 100],
                                   labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-100%'])

    denial_rates = (temp.groupby('minority_bin', observed=True)
                        .apply(lambda g: (g[TARGET] == 'Application_denied').mean() * 100)
                        .reset_index(name='denial_rate'))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(denial_rates['minority_bin'].astype(str),
            denial_rates['denial_rate'],
            marker='o', color='#e74c3c', linewidth=2)

    ax.set_title('Denial Rate vs Tract Minority Population %')
    ax.set_xlabel('Tract Minority Population')
    ax.set_ylabel('Denial Rate (%)')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    num_features = ['loan_amount', 'loan_to_value_ratio', 'interest_rate',
                    'rate_spread', 'total_loan_costs', 'origination_charges',
                    'loan_term', 'property_value', 'income', 'debt_to_income_ratio',
                    'tract_minority_population_percent',
                    'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage']

    existing = [c for c in num_features if c in df.columns]

    temp = df[existing].copy()
    temp['rate_spread'] = temp['rate_spread'].clip(-10, 10)
    temp['property_value'] = temp['property_value'].clip(upper=1_000_000)
    temp['income'] = temp['income'].clip(0, 500)
    temp['loan_amount'] = temp['loan_amount'].clip(upper=1_000_000)

    corr = temp.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.4, ax=ax,
                annot_kws={'size': 7})
    ax.set_title('Correlation Matrix - Numeric Features')
    plt.tight_layout()
    plt.show()

def runner(df):
    plot_target_distribution(df)
    plot_boxplots_numeric(df)
    plot_ltv_histogram(df)
    plot_scatter_income_loan(df)
    plot_loantype_vs_outcome(df)
    plot_denial_by_race(df)
    plot_preapproval_vs_outcome(df)
    plot_neighborhood_income(df)
    plot_minority_tract_denial(df)
    plot_correlation_heatmap(df)