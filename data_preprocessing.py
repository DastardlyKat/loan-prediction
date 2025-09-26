import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/swaksharbora/Documents/ML Projects/Loan Default Prediction/data/loan_default_dataset.csv')

# print(df.head())

# missing_values = df.isnull().sum()
# print("Missing Values: \n", missing_values)

df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].replace({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3})
df['Education'] = df['Education'].replace({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].replace({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].replace({'Urban': 1, 'Semiurban': 2, 'Rural': 3})
df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})

df['Gender'].fillna(df['Gender'].median(), inplace=True)
df['Married'].fillna(df['Married'].median(), inplace=True)
df['Dependents'].fillna(df['Dependents'].median(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].median(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)

# missing_values_after = df.isnull().sum()
# print("Missing Values: \n", missing_values_after)

X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)