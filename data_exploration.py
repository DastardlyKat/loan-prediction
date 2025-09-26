import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

df = pd.read_csv('/Users/swaksharbora/Documents/ML Projects/Loan Default Prediction/data/loan_default_dataset.csv')

print(df.head())

missing_values = df.isnull().sum()
print("Missing Values: \n", missing_values)

a = df['Loan_Status'].value_counts(1)
print(a)

b = df['Loan_Status'].value_counts(0)
print(b)