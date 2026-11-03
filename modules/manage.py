import pandas as pd
import numpy as np
from modules.audit import auditor, summarizer
from cleaner import one_value, cardinals, leaker, high_missing, transformer, filler
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
copy = filler(copy)

copy = mapper(copy)

print(summarizer(copy).to_string())