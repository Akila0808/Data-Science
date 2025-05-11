import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Setting seaborn style
sns.set(style="whitegrid")

# Creating output directory for plots
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# Loading the dataset
print("Loading dataset...")
df = pd.read_csv('accident.csv')

# Creating binary target variable: Is_Severe (1 if Number_of_Deaths > 0, else 0)
df['Is_Severe'] = (df['Number_of_Deaths'] > 0).astype(int)

# 1. Dataset Overview
print("\n=== Dataset Overview ===")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Checking unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
for col in ['State', 'Reason', 'Road_Type', 'Weather_Conditions', 'Road_Conditions', 'Alcohol_Involved', 'Driver_Fatigue']:
    print(f"{col}: {df[col].nunique()} unique values")

# Summary statistics
print("\nSummary Statistics for Numeric Columns:\n", df.describe())

# 2. Univariate Analysis
# Numeric Variables
print("\n=== Univariate Analysis ===")
plt.figure(figsize=(12, 4))

# Number of Deaths
plt.subplot(1, 3, 1)
sns.histplot(df['Number_of_Deaths'], bins=10, kde=True)
plt.title('Distribution of Number of Deaths')
plt.xlabel('Number of Deaths')
plt.ylabel('Count')

# Number of Injuries
plt.subplot(1, 3, 2)
sns.histplot(df['Number_of_Injuries'], bins=10, kde=True)
plt.title('Distribution of Number of Injuries')
plt.xlabel('Number of Injuries')
plt.ylabel('Count')

# Speed Limit
plt.subplot(1, 3, 3)
sns.histplot(df['Speed_Limit'], bins=10, kde=True)
plt.title('Distribution of Speed Limit')
plt.xlabel('Speed Limit (km/h)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('eda_plots/numeric_distributions.png')
plt.close()

# Categorical Variables
categorical_cols = ['State', 'Reason', 'Road_Type', 'Weather_Conditions', 'Road_Conditions', 'Alcohol_Involved', 'Driver_Fatigue']
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'eda_plots/{col}_distribution.png')
    plt.close()

# 3. Bivariate Analysis
print("\n=== Bivariate Analysis ===")
# Speed Limit vs. Is_Severe
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Is_Severe', y='Speed_Limit')
plt.title('Speed Limit vs. Accident Severity')
plt.xlabel('Is Severe (1 = Yes, 0 = No)')
plt.ylabel('Speed Limit (km/h)')
plt.savefig('eda_plots/speed_limit_vs_severity.png')
plt.close()

# Categorical Variables vs. Is_Severe
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=col, y='Is_Severe', order=df[col].value_counts().index)
    plt.title(f'Proportion of Severe Accidents by {col}')
    plt.xlabel(col)
    plt.ylabel('Proportion Severe')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'eda_plots/{col}_vs_severity.png')
    plt.close()

# Correlation Heatmap for Numeric Variables
plt.figure(figsize=(8, 6))
# Ensure only numeric columns are used for correlation
numeric_cols = ['Number_of_Deaths', 'Number_of_Injuries', 'Speed_Limit', 'Is_Severe']
# Remove any non-numeric columns like Road_Type
numeric_df = df[numeric_cols].select_dtypes(include=np.number)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Variables')
plt.savefig('eda_plots/correlation_heatmap.png')
plt.close()

# 4. Insights
print("\n=== Key Insights ===")
# Most common reasons for accidents
top_reasons = df['Reason'].value_counts().head(3)
print("Top 3 Reasons for Accidents:\n", top_reasons)

# States with highest severe accidents
severe_by_state = df.groupby('State')['Is_Severe'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 States by Proportion of Severe Accidents:\n", severe_by_state)

# Impact of Alcohol and Fatigue
alcohol_severe = df.groupby('Alcohol_Involved')['Is_Severe'].mean()
fatigue_severe = df.groupby('Driver_Fatigue')['Is_Severe'].mean()
print("\nProportion of Severe Accidents by Alcohol Involvement:\n", alcohol_severe)
print("\nProportion of Severe Accidents by Driver Fatigue:\n", fatigue_severe)

# Weather conditions and severity
weather_severe = df.groupby('Weather_Conditions')['Is_Severe'].mean().sort_values(ascending=False)
print("\nProportion of Severe Accidents by Weather Conditions:\n", weather_severe)

# Road conditions and severity
road_cond_severe = df.groupby('Road_Conditions')['Is_Severe'].mean().sort_values(ascending=False)
print("\nProportion of Severe Accidents by Road Conditions:\n", road_cond_severe)

print("\nEDA plots saved in 'eda_plots' directory.")