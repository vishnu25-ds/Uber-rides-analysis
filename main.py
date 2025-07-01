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


sns.countplot(x=df["purpose"])
plt.title("Value Counts of Feature: PURPOSE")
plt.xticks(rotation=90)


sns.countplot(x=df["category"])
plt.title("Value Counts of Feature: CATEGORY")
plt.xticks(rotation=0)


sns.countplot(x=df["day_name"])
plt.title("Value Counts of Feature: DAY NAME")
plt.xticks(rotation=45)


sns.countplot(x=df["time_label"])
plt.title("Value Counts of Feature: TIME LABEL")
plt.xticks(rotation=45)


sns.countplot(x=df["month"])
plt.title("Value Counts of Feature: MONTH")
plt.xticks(rotation=90)






import matplotlib.pyplot as plt
import seaborn as sns


top_values = df["start"].value_counts().head(10).index
sns.countplot(x=df[df["start"].isin(top_values)]["start"])
plt.title("Top 10 Value Counts of Feature: start")
plt.xticks(rotation=90)
plt.show()


sns.barplot(x= df["purpose"], y= df['miles'],hue = df["category"])
plt.title("Purpose VS Miles Travelled By Category ")
plt.xticks(rotation=90)


sns.boxplot(x=df["time_label"],y=df["miles"])
plt.title("Outliers in MILES over TIME")


sns.boxplot(x=df["duration"],data=df)
plt.title("Outliers in Duration")


sns.boxplot(x=df['miles'],data = df)
plt.title("Outliers in Miles")


sns.distplot(df["miles"])
plt.title("Density PLot for Miles")



sns.distplot(df["duration"])
plt.title("Density PLot for Duration")


sns.distplot(df["miles"],hist = False,label="Miles")
sns.distplot(df["duration"],hist = False, label="Duration")
plt.xlabel("Miles VS Duration")
plt.legend()


print(f"{Fore.CYAN}Outliers in data: \n.{Style.RESET_ALL}")
df = df[df['start_date'] != "Totals"]
df['start_date'] = pd.to_datetime(df['start_date'])
df.set_index('start_date', inplace=True)
daily_miles = df.resample('D')['miles'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_miles.index, daily_miles.values)
plt.xlabel('Date')
plt.ylabel('Miles Driven')
plt.title('Miles Driven Over Time')
plt.show()



# Calculate IQR for 'miles' feature
Q1_miles = np.percentile(df['miles'], 25)
Q3_miles = np.percentile(df['miles'], 75)
IQR_miles = Q3_miles - Q1_miles

# Calculate IQR for 'duration' feature
Q1_duration = np.percentile(df['duration'], 25)
Q3_duration = np.percentile(df['duration'], 75)
IQR_duration = Q3_duration - Q1_duration

# Define outlier boundaries
lower_bound_miles = Q1_miles - 1.5 * IQR_miles
upper_bound_miles = Q3_miles + 1.5 * IQR_miles

lower_bound_duration = Q1_duration - 1.5 * IQR_duration
upper_bound_duration = Q3_duration + 1.5 * IQR_duration


# Identify and deal with outliers
df['miles'] = np.where(df['miles'] < lower_bound_miles, lower_bound_miles, df['miles'])
df['miles'] = np.where(df['miles'] > upper_bound_miles, upper_bound_miles, df['miles'])

df['duration'] = np.where(df['duration'] < lower_bound_duration, lower_bound_duration, df['duration'])
df['duration'] = np.where(df['duration'] > upper_bound_duration, upper_bound_duration, df['duration'])



print(f"{Fore.RED}After Outliers Treatment: \n.{Style.RESET_ALL}")
daily_miles = df.resample('D')['miles'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_miles.index, daily_miles.values)
plt.xlabel('Date')
plt.ylabel('Miles Driven')
plt.title('Miles Driven Over Time(After Outliers Treatment)')
plt.show()



data.head()





# Using pandas to encode featuers 
df_encoded = pd.get_dummies(data, columns=['category', 'start',"purpose", 'stop', 'day_name', 'time_label', 'month'])



# Spliting Data into Train and Test
from sklearn.model_selection import train_test_split
X = df_encoded.drop(['miles',"start_date","end_date"], axis=1) 
y = df_encoded['miles']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'Gradient Boosting Regressor' : GradientBoostingRegressor(),
    'ADA Boost' : AdaBoostRegressor(),
    'Linear SVR' : LinearSVR(),
    'KNN Regressor' : KNeighborsRegressor(),
    'Cat Boost Regressor' : CatBoostRegressor()
}


accuracy = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred



for name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy[name] = r2
    
    print(f"Results for {name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    plt.figure(figsize=(10, 2))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Trend')
    plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Trend')
    plt.xlabel('Data Index')
    plt.ylabel('Trend')
    plt.title(f'{name}: Actual Trend vs. Predicted Trend')
    plt.legend()
    plt.show()
    print()



from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Define a threshold for classification (need to adjust this based on the problem)
threshold = 50  # 

# Convert regression predictions to binary classification labels
binary_predictions = {name: np.where(y_pred > threshold, 1, 0) for name, y_pred in predictions.items()}

precision = {}
recall = {}
f1 = {}

for name, y_pred_bin in binary_predictions.items():
    precision[name] = precision_score(y_test > threshold, y_pred_bin)
    recall[name] = recall_score(y_test > threshold, y_pred_bin)
    f1[name] = f1_score(y_test > threshold, y_pred_bin)

for name in precision.keys():
    print(f"Results for {name}:")
    print(f"Precision: {precision[name]:.4f}")
    print(f"Recall: {recall[name]:.4f}")
    print(f"F1 Score: {f1[name]:.4f}")
    print()





import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Initialize lists to store metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Create an empty dictionary to store model predictions
temp_predictions = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    temp_predictions[model_name] = y_pred

for model_name in temp_predictions.keys():
    y_pred_bin = np.where(temp_predictions[model_name] > threshold, 1, 0)
    
    accuracy_value = accuracy_score(y_test > threshold, y_pred_bin)
    precision_value = precision_score(y_test > threshold, y_pred_bin)
    recall_value = recall_score(y_test > threshold, y_pred_bin)
    f1_value = f1_score(y_test > threshold, y_pred_bin)

    # Append metrics to the lists
    model_names.append(model_name)
    accuracies.append(accuracy_value)
    precisions.append(precision_value)
    recalls.append(recall_value)
    f1_scores.append(f1_value)

# Create the metrics DataFrame
metrics_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})



# Display the metrics DataFrame
print(metrics_df)

import matplotlib.pyplot as plt

# Plot the metrics using a bar graph
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2
bar_positions = np.arange(len(metrics_df['Model']))

for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
    ax.bar(bar_positions + i * bar_width, metrics_df[metric], width=bar_width, label=metric)

ax.set_title('Model Performance Metrics')
ax.set_ylabel('Score')
ax.set_xlabel('Model')
ax.set_xticks(bar_positions + bar_width * 1.5)
ax.set_xticklabels(metrics_df['Model'], rotation=90)
ax.legend()

plt.show()

