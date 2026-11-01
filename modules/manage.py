import time

import pandas as pd
import numpy as np
from modules.audit import auditor, summary
from cleaner import one_value, cardinals, leaker, high_missing, transformer
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


# print(summary(copy).to_string())
