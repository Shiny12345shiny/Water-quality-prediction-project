import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""This file is responsible for data gathering and splitting the data into training and testing sets."""
def data_cleaning(df):
    df = df.fillna(df.median())
   
    for column in df.columns[:-1]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column]=np.where(df[column]<lower_bound,lower_bound,df[column])
        df[column]=np.where(df[column]>upper_bound,upper_bound,df[column])
    y=df['Potability'].copy()
    X=df.drop('Potability',axis=1).copy()
    from pathlib import Path
    p=Path("processed_data/processed_label.csv")
    X.to_csv(p,index=False)
    p=Path("processed_data/processed_class.csv")
    y.to_csv(p,index=False)
    return X,y


   