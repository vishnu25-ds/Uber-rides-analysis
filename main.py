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




# Try parsing with the first format
try:
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%m/%d/%Y %H:%M')
except ValueError:
    # If the first format fails, try parsing with the second format
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%m-%d-%Y %H:%M')

# Repeat the same for 'END_DATE'
try:
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], format='%m/%d/%Y %H:%M')
except ValueError:
    df['END_DATE'] = pd.to_datetime(df['END_DATE'], format='%m-%d-%Y %H:%M')



df.head()

df.rename(columns={
    'START_DATE': 'start_date',
    'END_DATE': 'end_date',
    'CATEGORY': 'category',
    'START' : 'start',
    'STOP' : 'stop',
    'MILES' : 'miles',
    'PURPOSE' : 'purpose'
}, inplace=True)


df.head()



# Creating new features for better analysis and accuracy

df['day_name'] = df['start_date'].dt.day_name()

time_periods = [0, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']

df['time_label'] = pd.cut(df['start_date'].dt.hour, bins=time_periods, labels=labels, right=False)

df["month"] = df['start_date'].dt.month_name()

df['duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 60

print(f"{Fore.BLUE}New Data: \n.{Style.RESET_ALL}")
data = df.copy()
df.head(5)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fast_eda(df):
    # Exclude non-numeric columns from correlation computation
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_columns]

    # Display heatmap
    if numeric_df.columns.nunique() < 15:
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Spectral", linewidths=2, linecolor="#000000", fmt='.2f')
        plt.show()
    elif numeric_df.columns.nunique() < 25:
        
        pass

fast_eda(df)

df.describe()


popular_destinations = df['start'].value_counts().head(10)

# Plot the top 10 popular destinations
plt.figure(figsize=(12, 6))
sns.barplot(x=popular_destinations.values, y=popular_destinations.index, palette='viridis')
plt.title('Popular Pickup locations')
plt.xlabel('Number of Rides')
plt.ylabel('Pickup locations')
plt.show()



popular_destinations = df['stop'].value_counts().head(10)

# Plot the top 10 popular destinations
plt.figure(figsize=(12, 6))
sns.barplot(x=popular_destinations.values, y=popular_destinations.index, palette='viridis')
plt.title('Popular Drop locations')
plt.xlabel('Number of Rides')
plt.ylabel('Destination')
plt.show()

