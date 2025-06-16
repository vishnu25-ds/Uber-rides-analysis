import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
from fasteda import fast_eda

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("UberDataset.csv")
df


print(f"{Fore.BLUE}The Shape of Dataset is: {df.shape}.{Style.RESET_ALL}")


print(f"{Fore.BLUE}Information of Features in Dataset: \n.{Style.RESET_ALL}")
df.info()

# Try to parse 'START_DATE' and 'END_DATE' with multiple formats
df['START_DATE'] = pd.to_datetime(df['START_DATE'], infer_datetime_format=True, errors='coerce')
df['END_DATE'] = pd.to_datetime(df['END_DATE'], infer_datetime_format=True, errors='coerce')

print(f"{Fore.BLUE}Information of Features in Dataset(after): \n.{Style.RESET_ALL}")
df.info()

print(f"{Fore.RED}Statistical Analysis of Feature 'MILES: \n.{Style.RESET_ALL}")
df.describe()

print(f"{Fore.YELLOW}Duplicat Row in Dataset: \n.{Style.RESET_ALL}")

df[df.duplicated()]



df.drop_duplicates(inplace = True)


print(f"{Fore.MAGENTA}Null Values in Each Feature (before): \n.{Style.RESET_ALL}")
df.isnull().sum()


df["PURPOSE"] = df["PURPOSE"].fillna("Missing")
df.dropna(inplace = True)


print(f"{Fore.MAGENTA}Null Values in Each Feature (after): \n.{Style.RESET_ALL}")
df.isnull().sum()

