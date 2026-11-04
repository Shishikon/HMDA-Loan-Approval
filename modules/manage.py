import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from modules.audit import auditor, summarizer
from cleaner import one_value, cardinals, leaker, high_missing, transformer, filler_fit, filler_transform
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


train, test = train_test_split(copy, test_size=0.2, random_state=42, stratify=copy['action_taken'])


stats = filler_fit(train)
train = filler_transform(train, stats)
test = filler_transform(test, stats)


y_train = train['action_taken']
y_test = test['action_taken']


X_train = train.drop(columns=['action_taken'])
X_test = test.drop(columns=['action_taken'])



from EDA import *


plot_correlation_heatmap(train)
# print(copy['tract_minority_population_percent'].head(3).to_string())
# print(copy['tract_minority_population_percent'].value_counts().to_string())
