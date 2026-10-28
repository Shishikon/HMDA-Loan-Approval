import pandas as pd
from certifi import __main__
from audit import *




raw = pd.read_csv('../actions_taken_1-2-3_loan_purposes_1_state_PA.csv', low_memory=False)

copy = raw.copy()





