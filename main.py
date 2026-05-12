from sqlite3 import DateFromTicks
from src.data_preprocessing import data_cleaning
import pandas as pd
from src.imbalance import data_balance
from sklearn.ensemble import RandomForestClassifier
from src.model import algo

df=pd.read_csv('water_potability (1).csv')
print(df)
cleaned_df=data_cleaning(df)
print(cleaned_df)
balanced_data=data_balance()

algo(RandomForestClassifier())