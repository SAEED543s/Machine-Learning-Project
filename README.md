# Employee Turnover Prediction

This project uses **Machine Learning** to predict whether an employee will leave the company (turnover) based on their data.  
We train and evaluate two models: **Decision Tree** and **Random Forest**.

---

## 📂 Project Structure
project/
│── employee_data.csv # Dataset
│── main.py # Project code
│── images/ # Saved plots
│ ├── correlation_heatmap.png
│ ├── model_comparison.png
│ ├── confusion_matrices.png
│── requirements.txt # Dependencies
│── README.md # Documentation

---

## 📊 Dataset
The dataset contains employee information such as:
- Age  
- Job Role  
- Years at Company  
- Salary  
- Performance Rating  
- And the target column **`quit`** (1 = Employee left, 0 = Employee stayed)  

---

## 🚀 Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/username/employee-turnover-prediction.git
cd employee-turnover-prediction

2.Install the required dependencies:
pip install -r requirements.txt
3.Run the project
python main.py
📷 Sample Outputs
🔥 Correlation Heatmap

📊 Model Comparison

📌 Confusion Matrices
🛠️ Technologies Used
Python 3.x

. Pandas

. NumPy

. Matplotlib

. Seaborn

. Scikit-learn

. Imbalanced-learn

. SMOTE
📈 Models Used
. Decision Tree

. Random Forest
📌 Results
The Random Forest model generally performed better in terms of accuracy, precision, recall, and F1-score compared to the Decision Tree.

📧 saeedismaileldad@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/saeed-ismail-el-dad-82a63a361?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)




********** Code  **************



# Importing Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , precision_score , recall_score , f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree , export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from yellowbrick.target import ClassBalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from ipywidgets import  interact, interactive, IntSlider, FloatSlider, Dropdown, Checkbox
from IPython.display import display, clear_output
INTERACTIVE_AVAILABLE = True

np.random.seed(42)

print('Libraries imported successfully!')
print('='*35)

"""# Uploding Data &  Exploratory Data Analysis"""

df = pd.read_csv('/content/employee_data.csv')

df.head()

df.tail()

df.info()

print(df.shape)

df.dtypes

df.columns.tolist()

df.columns.T

"""# Missing Data"""

missing_data = df.isnull().sum()
missing_data

missing_percent  = (missing_data/len(df))*100
 missing_percent

df.processed = df.copy()

"""# Handling Missing Values"""

columns_to_drop = missing_percent[missing_percent > 50].index.tolist()
if columns_to_drop:
    print(f"a) Dropping columns with >50% missing_data: {columns_to_drop}")
    df.processed = df.processed.drop(columns=columns_to_drop)
else:
    print("a) No columns to drop (none have >50% missing_data)")

num_columns = df.processed.select_dtypes(include=[np.number]).columns
for col in num_columns:
    if df.processed[col].isnull().sum() ==0:
        median_data = df.processed[col].median()
        df.processed[col].fillna(median_data, inplace=True)
        print(f"b) Filled {col} missing data with median: {median_data:.2f}")

categorical_columns = df.processed.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df.processed[col].isnull().sum() > 0:
        mode_data = df.processed[col].mode()[0] if not df.processed[col].mode().empty else 'Unknown'
        df.processed[col].fillna(mode_data, inplace=True)
        print(f"c) Filled {col} missing data with mode: {mode_data}")

missing_data_after = df.processed.isnull().sum()
missing_data_after

"""# Encoding Categorical Variables"""

binary_columns = []
for col in categorical_columns:
    if col in df.processed.columns and df.processed[col].nunique() == 2:
        binary_columns.append(col)
        le = LabelEncoder()
        df.processed[f'{col}_encoded'] = le.fit_transform(df.processed[col])
        print(f") Label encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

multi_cat_columns = []
for col in categorical_columns:
    if col in df.processed.columns and df.processed[col].nunique() > 2 and df.processed[col].nunique() <= 10:
        multi_cat_columns.append(col)

if multi_cat_columns:
    print(f"b) Applying One-Hot Encoding to: {multi_cat_columns}")
    df_encoded = pd.get_dummies(df.processed, columns=multi_cat_columns, prefix=multi_cat_columns)
    df.processed = df_encoded
else:
    print("b)  No multi-category columns found for One-Hot Encoding")

"""# Feature Scaling"""

numeric_features = df.processed.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if not col.endswith('_encoded') and col != 'survived']

scaler_std = StandardScaler()
df_std_scaled = df.processed.copy()
df_std_scaled[numeric_features] = scaler_std.fit_transform(df.processed[numeric_features])
print(f"a) Applied StandardScaler to: {numeric_features}")

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
df_minmax_scaled = df.processed.copy()
df_minmax_scaled[numeric_features] = scaler_minmax.fit_transform(df.processed[numeric_features])
print(f"b)  Applied MinMaxScaler to: {numeric_features}")

"""# Comparison of scaling methods"""

comparison_col = numeric_features[0] if numeric_features else None
if comparison_col:
      print(f"Original {comparison_col} - Mean: {df.processed[comparison_col].mean():.2f}, Std: {df.processed[comparison_col].std():.2f}")
      print(f"StandardScaler {comparison_col} - Mean: {df_std_scaled[comparison_col].mean():.2f}, Std: {df_std_scaled[comparison_col].std():.2f}")
      print(f"MinMaxScaler {comparison_col} - Min: {df_minmax_scaled[comparison_col].min():.2f}, Max: {df_minmax_scaled[comparison_col].max():.2f}")
else:
 print("No numeric features found for scaling")

"""# OUTLIER DETECTION"""

outlier_cols = [col for col in df.processed.select_dtypes(include=[np.number]).columns
                if col not in ['survived'] and not col.endswith('_encoded')]

if outlier_cols:

    print("a) Boxplot Visualization:")
    n_cols = len(outlier_cols)
    fig, axes = plt.subplots(nrows=(n_cols+2)//3, ncols=3, figsize=(15, 5*((n_cols+2)//3)))
    axes = axes.flatten() if n_cols > 1 else [axes]

for i, col in enumerate(outlier_cols):
        if i < len(axes):
            df.processed.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()

"""# IQR Method"""

outliers_iqr = {}

for col in outlier_cols:
      Q1 = df.processed[col].quantile(0.25)
      Q3 = df.processed[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR

      outliers = df.processed[(df.processed[col] < lower_bound) | (df.processed[col] > upper_bound)]
      outliers_iqr[col] = len(outliers)
      print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df.processed)*100:.1f}%)")
      print(f"      Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df.processed)*100:.1f}%)")
print(f"      Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

"""# Z-Score Method"""

from scipy import stats

outliers_zscore = {}
for col in outlier_cols:
     z_scores = np.abs(stats.zscore(df.processed[col].dropna()))
     outliers = len(z_scores[z_scores > 3])
     outliers_zscore[col] = outliers
     print(f"   {col}: {outliers} outliers with |z-score| > 3 ({outliers/len(df.processed)*100:.1f}%)")

"""# Outlier Treatment"""

df_no_outliers = df.processed.copy()

for col in outlier_cols:
    if outliers_iqr.get(col, 0) > 0:
        Q1 = df.processed[col].quantile(0.25)
        Q3 = df.processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"   Capped outliers in {col} to range [{lower_bound:.2f}, {upper_bound:.2f}]")

out_indices = outliers_iqr
out_indices.keys()

"""# EDA"""

df.describe().T

df.describe(include='all').T

for col in df.columns:
  print(col,df[col].unique())

for col in df.columns:
    print(df[col].value_counts)

#for col in df.columns:
 #print(f"{col}: {df[col].unique}")

df.value_counts()

df.duplicated().sum()

num_columns =df.select_dtypes(include=[np.number]).columns
num_columns

num_columns =df.select_dtypes(include=[np.number]).columns.tolist()
num_columns

print(df[num_columns].describe)

num_stats = df.describe()
num_stats

num_stats = df.describe().T
num_stats

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns

categorical_columns = df.select_dtypes(include=['object']).columns.T
categorical_columns
for col in categorical_columns:
  print(df[col].unique().T)

df.describe(include='all').T


df.select_dtypes(include='object').info()

for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")

categorical_columns = df.select_dtypes(include=['object']).columns.T
categorical_columns
for col in categorical_columns:
  print(df[col].value_counts().head())

for col in num_columns:
    print(f"\n{col}:")
    print(f" : {df[col].mean():.2f}")
    print(f" : {df[col].median():.2f}")
    print(f" : {df[col].min():.2f}")
    print(f" : {df[col].max():.2f}")
    print(f" : {df[col].quantile(0.25):.2f}")
    print(f" : {df[col].quantile(0.50):.2f}")
    print(f" : {df[col].quantile(0.75):.2f}")
    print(f" : {df[col].skew():.2f}")
    print(f" : {df[col].kurtosis():.2f}")
    print(f" : {df[col].sum():.2f}")
    print(f" : {df[col].nunique():.2f}")
    print(f" : {df[col].count():.2f}")
    print(f" : {df[col].mode().iloc[0]:.2f}")
    print(f" : {df[col].std():.2f}")
    print(f" : {df[col].var():.2f}")
    print(f" : {(df[col].std()/df[col].mean())*100:.2f}%")

sns.pairplot(df)

sns.heatmap(df.processed.corr(), annot=True)

sns.countplot(df['quit'])

sns.countplot(x='quit', hue='last_evaluation', data=df)

sns.countplot(x='quit', hue='number_project', data=df)

sns.countplot(x='quit', hue='average_montly_hours', data=df)

sns.countplot(x='quit', hue='time_spend_company', data=df)

sns.countplot(x='quit', hue='Work_accident', data=df)

sns.countplot(x='quit', hue='promotion_last_5years', data=df)

sns.countplot(x='quit', hue='salary', data=df)

sns.countplot(x='quit', hue='department', data=df)

sns.countplot(x='quit', hue='satisfaction_level', data=df)

sns.histplot(df['last_evaluation'])

sns.histplot(df['number_project'])

sns.histplot(df['average_montly_hours'])

sns.histplot(df['time_spend_company'])

sns.histplot(df['Work_accident'])

sns.histplot(df['promotion_last_5years'])

sns.histplot(df['salary'])

sns.histplot(df['department'])

sns.histplot(df['quit'])

sns.histplot(df['satisfaction_level'])

sns.boxplot(data = df , y = "last_evaluation"  , x = "salary")

total_without_out = df["last_evaluation"].drop(out_indices["last_evaluation"] )

sns.boxplot(total_without_out)

fig , axes    = plt.subplots(4 , 3 , figsize=(12,15))

sns.countplot(x = "department"  , hue = "salary" , data = df , ax = axes[0,0] )
axes[0,0].set(ylim = (0,500))
sns.countplot(x = "department"  , hue = "last_evaluation" , data = df , ax = axes[0,1] )
axes[0,1].set(ylim = (0,500))
axes[1,0].set(title = "department when salary = 1"  , ylim = (0,100))
sns.countplot(x = "department"  , hue = "number_project" , data = df , ax = axes[1,0] )
axes[1,1].set(title = "department when salary = 0" ,ylim = (0,100) )
sns.countplot(x = "department"  , hue = "average_montly_hours" , data = df , ax = axes[1,1] )
axes[2,0].set(title = "department when salary = 1"  , ylim = (0,100))
sns.countplot(x = "department"  , hue = "time_spend_company" , data = df , ax = axes[2,0] )
axes[2,1].set(title = "department when salary = 0" ,ylim = (0,100) )
sns.countplot(x = "department"  , hue = "Work_accident" , data = df , ax = axes[2,1] )
axes[3,0].set(title = "department when salary = 1"  , ylim = (0,100))
sns.countplot(x = "department"  , hue = "promotion_last_5years" , data = df , ax = axes[3,0] )
axes[3,1].set(title = "department when salary = 0" ,ylim = (0,100) )
sns.countplot(x = "department"  , hue = "satisfaction_level" , data = df , ax = axes[3,1] )
sns.histplot(df["last_evaluation"] , ax = axes[0,2])
sns.histplot(df["number_project"] , ax = axes[1,2])
sns.histplot(df["average_montly_hours"] , ax = axes[2,2])
sns.histplot(df["time_spend_company"] , ax = axes[3,2])

"""## Distribution Analysis"""

processed_num_stats = df.processed.describe().T
required_stats = ['mean', 'std']

for col in processed_num_stats.index:
    if col in df.processed.columns and all(stat in processed_num_stats.index for stat in required_stats):
        mean_val = processed_num_stats.loc['mean', col]
        std_val = processed_num_stats.loc['std', col]
        skewness = df.processed[col].skew()
        kurtosis = df.processed[col].kurtosis()

        print(f"\n   {col}:")
        print(f"      Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"      Skewness: {skewness:.2f} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Approximately normal'})")
        print(f"      Kurtosis: {kurtosis:.2f} ({'Heavy-tailed' if kurtosis > 0 else 'Light-tailed'})")
    elif col not in df.processed.columns:
        print(f"\n   Skipping {col}: Column not found in df.processed")
    else:
        print(f"\n   Skipping {col}: Required statistics ('mean', 'std') not found in processed_num_stats index")

"""# Data Visualizations

**a) Histogram**
"""

numeric_cols = df.processed.select_dtypes(include=[np.number]).columns[:5]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):

        df.processed[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

numeric_cols = df.processed.select_dtypes(include=[np.number]).columns[:5]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i <4:

        df.processed[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

numeric_cols = df.processed.select_dtypes(include=[np.number]).columns[:5]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):

        df.processed[col].hist(bins=30, ax=axes[i], alpha=0.2, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

global_mean = df["last_evaluation"].mean()
global_mean

inter_mean = df.groupby("department")["last_evaluation"].mean()
inter_mean

risk_ratio = inter_mean/global_mean
risk_ratio

"""# Task 3) Encode Categorical Features"""

salary_dummies = pd.get_dummies(df['salary'], prefix='Salary')
print(f"\nSalary dummy variables:")
print(salary_dummies.head())

dept_dummies = pd.get_dummies(df['department'], prefix='department', prefix_sep='_')
df_encoded_pandas = pd.concat([df.drop(['department', 'salary'], axis=1),
                              dept_dummies, salary_dummies], axis=1)

df_encoded_pandas.head()

df_encoded_pandas.shape

from sklearn.preprocessing import OneHotEncoder , LabelEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')

encoded_df = encoder.fit_transform(df[['department', 'salary']])

feature_names = encoder.get_feature_names_out()

df_encoded_sklearn = pd.DataFrame(encoded_df, columns=feature_names)

df_final_sklearn = pd.concat([df.drop(['department', 'salary'], axis=1).reset_index(drop=True),
                             df_encoded_sklearn], axis=1)

df_final_sklearn.head()

df_final_sklearn.shape

"""# Task 4 ) Visualize Class Imbalance"""

df.dropna(inplace=True)

X = df.drop('quit', axis=1)
y = df['quit']

X = pd.get_dummies(X, drop_first=True)

print(df.shape)

print(X.shape)

print(y.shape)

class_counts = y.value_counts()
class_counts

from collections import Counter

class_counts = Counter(y)
total_samples = len(y)
class_imbalance = {class_label: count / total_samples for class_label, count in class_counts.items()}

for class_label, count in class_counts.items():
    percentage = (count / total_samples) * 100
    print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

majority_class = max(class_counts.values())
minority_class = min(class_counts.values())
imbalance_ratio = majority_class / minority_class

print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

print(f"Minority class percentage: {(minority_class/total_samples)*100:.2f}%")

print(f"Majority class percentage: {(majority_class/total_samples)*100:.2f}%")

if imbalance_ratio > 1.5:
    imbalance_severity = "MODERATE" if imbalance_ratio <= 4 else "SEVERE" if imbalance_ratio <= 9 else "EXTREME"
    print(f"Class imbalance detected: {imbalance_severity} imbalance")
    needs_sampling = True
else:
    print("No significant class imbalance detected")
    needs_sampling = False

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Class Imbalance Analysis', fontsize=16, fontweight='bold')
visualizer = ClassBalance(labels=['Stayed', 'Left'])
visualizer.fit(y)
visualizer.show(ax=axes[0, 0])
axes[0, 0].set_title('Yellowbrick Class Balance Visualizer')
axes[0, 1].bar(class_counts.keys(), class_counts.values(),
               color=['skyblue', 'salmon'], alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Class Frequency Distribution')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['Stayed (0)', 'Left (1)'])
for i, (class_label, count) in enumerate(class_counts.items()):
    axes[0, 1].text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')

"""
Pie chart
"""

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
labels = ['minority_class', 'majority_class']
sizes = list(class_counts.values())
colors = ['lightblue', 'lightcoral']
explode = (0.05, 0.1)
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True, textprops={'fontsize': 9})

ax1.set_title('Pie Chart', fontsize=12, fontweight='bold', pad=20)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2 = plt.subplot(2, 3, 2)
bars = ax2.bar(range(len(class_counts)), list(class_counts.values()),
               color=['skyblue', 'salmon'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('total_samples')
ax2.set_ylabel('11582')
ax2.set_title('Bar Chart')
ax2.set_xticks(range(len(class_counts)))
ax2.set_xticklabels(['minority_class', 'majority_class'])
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold')

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

train_counts = Counter(y_train)
for class_label, count in train_counts.items():
    percentage = (count / len(y_train)) * 100
    print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

sampling_results = {}

"""Random Oversampling"""

ros = RandomOverSampler(random_state=42)
X_train_RandomOverSampler, y_train_RandomOverSampler = ros.fit_resample(X_train, y_train)
ros_counts = Counter(y_train_RandomOverSampler)
sampling_results['Random Oversampling'] = (X_train_RandomOverSampler, y_train_RandomOverSampler, ros_counts)

for class_label, count in ros_counts.items():
        percentage = (count / len(y_train_RandomOverSampler)) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

"""SMOTE"""

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_counts = Counter(y_train_smote)
sampling_results['SMOTE'] = (X_train_smote, y_train_smote, smote_counts)

for class_label, count in smote_counts.items():
        percentage = (count / len(y_train_smote)) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

"""Random Undersampling"""

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
rus_counts = Counter(y_train_rus)
sampling_results['Random Undersampling'] = (X_train_rus, y_train_rus, rus_counts)

for class_label, count in rus_counts.items():
        percentage = (count / len(y_train_rus)) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

"""SMOTE + Tomek"""

smote_tomek = SMOTETomek(random_state=42)
X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)
st_counts = Counter(y_train_st)
sampling_results['SMOTE + Tomek'] = (X_train_st, y_train_st, st_counts)

for class_label, count in st_counts.items():
        percentage = (count / len(y_train_st)) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

fig , axes = plt.subplots(2 , 2 , figsize = (15 , 10))
axes = axes.flatten()
for i , (name , (X_resampled , y_resampled , counts)) in enumerate(sampling_results.items()):
    axes[i].bar(counts.keys() , counts.values() , color = ['skyblue' , 'salmon'] , alpha = 0.8 , edgecolor = 'black' , linewidth = 1.5)
    axes[i].set_xlabel('Class')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{name} Sampling')
    axes[i].set_xticks([0 , 1])
    axes[i].set_xticklabels(['minority_class' , 'majority_class'])

"""# Task 5: Create Training and Validation Sets
    
"""

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

print(f"Training set: {X_train.shape[0]} ({(X_train.shape[0]/total_samples)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} ({(X_val.shape[0]/total_samples)*100:.1f}%)")

train_counts = Counter(y_train)
for class_label, count in train_counts.items():
    percentage = (count / len(y_train)) * 100
    class_name = 'minority_class'  if class_label  ==  1 else 'majority_class'
    print(f"{class_label} ({class_name}): {count} ({percentage:.2f}%)")

val_counts = Counter(y_val)
for class_label, count in val_counts.items():
    percentage = (count / len(y_val)) * 100
    class_name = 'minority_class' if class_label == 0 else 'majority_class'
    print(f"{class_label} ({class_name}): {count} ({percentage:.2f}%)")

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in sss.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

sss_train_counts = Counter(y_train)
sss_val_counts = Counter(y_val)

print(f"train_test_split - Training: {dict(train_counts)}")
print(f"StratifiedShuffleSplit - Training: {dict(sss_train_counts)}")
print(f"{train_counts == sss_train_counts}")

original_counts = Counter(df)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

fig, axes = plt.subplots(2, 2, figsize=(25, 20))
ax1 = axes[0, 0]
ax1.bar(range(len(original_counts)), list(original_counts.values()),
        color=colors[:len(original_counts)], alpha=0.6, edgecolor='black')
ax1.set_title('n\(df)')
ax1.set_xticks(range(len(original_counts)))
ax1.set_xticklabels(list(original_counts.keys()))
for i, v in enumerate(original_counts.values()):
    ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()
print("Original counts:", original_counts)
for i, count in enumerate(original_counts.values()):
    ax1.text(i, count + 20, str(count), ha='center', va='bottom', fontweight='bold')

for i, count in enumerate(train_counts.values()):
    ax2.text(i, count + 15, str(count), ha='center', va='bottom', fontweight='bold')
ax2 = axes[0, 1]
ax2.bar(range(len(train_counts)), list(train_counts.values()),
        color=colors, alpha=0.8, edgecolor='black')
ax2.set_title('Training Set\n(80% from data')
ax2.set_xticks(range(len(train_counts)))
ax2.set_xticklabels(labels)
ax2.set_ylabel('original_counts')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


ax1 = axes[0, 0]
ax1.bar(range(len(original_counts)), list(original_counts.values()),
        color=colors[:len(original_counts)], alpha=0.8, edgecolor='black')
ax1.set_title('\n(Original Data)')
ax1.set_xticks(range(len(original_counts)))

ax1.set_xticklabels(list(original_counts.keys()))

ax2 = axes[0, 1]
ax2.bar(range(len(train_counts)), list(train_counts.values()),
        color=colors[:len(train_counts)], alpha=0.8, edgecolor='black')
ax2.set_title('Training Set\n(80% from data)')
ax2.set_xticks(range(len(train_counts)))

ax2.set_xticklabels(list(train_counts.keys()))

ax3 = axes[0, 2]
ax3.bar(range(len(val_counts)), list(val_counts.values()),
        color=colors[:len(val_counts)], alpha=0.8, edgecolor='black')
ax3.set_title('Validation Set\n(20% from data)')
ax3.set_xticks(range(len(val_counts)))

ax3.set_xticklabels(list(val_counts.keys()))


ax4 = axes[1, 0]
ax4.pie(original_counts.values(), labels=list(original_counts.keys()), autopct='%1.1f%%',
        colors=colors[:len(original_counts)], startangle=90)
ax4.set_title('Original Data Distribution')

ax5 = axes[1, 1]

ax5.pie(train_counts.values(), labels=list(train_counts.keys()), autopct='%1.1f%%',
        colors=colors[:len(train_counts)], startangle=90)
ax5.set_title('Training Set Distribution')

ax6 = axes[1, 2]

val_labels = list(val_counts.keys())
explode_values = [0.05] * len(val_labels)
ax6.pie(val_counts.values(), labels=val_labels, autopct='%1.1f%%',
        colors=colors[:len(val_counts)], startangle=90, explode=explode_values)
ax6.set_title('Validation Set Distribution')

plt.tight_layout()
plt.show()

for i, feature in enumerate(X.columns, 1):
    print(f"{i:2d}. {feature}")

print(X_train.describe())

missing_train = X_train.isnull().sum()
print(f" missing data : {missing_train.sum()}")

missing_val = X_val.isnull().sum()
print(f"missing data :  {missing_val.sum()}")

train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('training_set.csv', index=False)

val_data = pd.concat([X_val, y_val], axis=1)
val_data.to_csv('validation_set.csv', index=False)

print(f"   • Total data: {total_samples}")
print(f"   • Training: {len(y_train)} from data({(len(y_train)/total_samples)*100:.1f}%)")
print(f"   • Validation: {len(y_val)} from data({(len(y_val)/total_samples)*100:.1f}%)")

"""#  Task 6 & 7: Build a Decision Tree Classifier with Interactive Controls"""

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

n_samples = 11581

ages = np.random.randint(22, 65, n_samples)

max_experience_per_person = np.minimum(40, ages - 20)  # Element-wise minimum
experience = np.array([np.random.randint(0, max_exp + 1) for max_exp in max_experience_per_person])

satisfaction = np.random.uniform(1, 10, n_samples)
salary_numeric = np.random.uniform(30000, 150000, n_samples)
work_hours = np.random.uniform(35, 60, n_samples)

departments = np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], n_samples)
education_levels = np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples,
                                  p=[0.4, 0.35, 0.15, 0.1])
gender = np.random.choice(['Male', 'Female'], n_samples)

df.shape

ages = np.random.randint(22, 65, n_samples)
max_experience_per_person = np.minimum(40, ages - 20)
experience = np.array([np.random.randint(0, max_exp + 1) for max_exp in max_experience_per_person])
satisfaction = np.random.uniform(1, 10, n_samples)
salary_numeric = np.random.uniform(30000, 150000, n_samples)
work_hours = np.random.uniform(35, 60, n_samples)

leave_probability = ((10 - satisfaction) * 0.1 + (work_hours - 40) * 0.02 +   (80000 - salary_numeric) * 0.000001  )
leave_probability = np.clip(leave_probability, 0, 1)
target = np.random.binomial(1, leave_probability)

salary_counts = Counter(df['salary'])
for class_label, count in salary_counts.items():
    percentage = (count / len(df)) * 100
    class_name = '*' if class_label == 0 else '*'
    print(f"  {class_name} ({class_label}): {count} ({percentage:.1f}%)")

df_encoded = pd.get_dummies(df.drop('salary', axis=1), drop_first=True)
X = df_encoded
y = df['salary']

print(f"Features بعد التحويل: {X.shape[1]}")

for i, feature in enumerate(X.columns, 1):
    print(f"  {i:2d}. {feature}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} data")
print(f"Validation set: {X_val.shape[0]} data")

def build_decision_tree(max_depth=5, min_samples_split=20, min_samples_leaf=10, criterion='gini', max_features='sqrt', random_state=42, show_tree=True, show_metrics=True):
   dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        random_state=random_state
    )

dt_classifier = DecisionTreeClassifier()

print("Classifier parameters:", dt_classifier.get_params())

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_train_pred = dt_classifier.predict(X_train)
y_val_pred = dt_classifier.predict(X_val)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

overfitting = train_accuracy - val_accuracy
print(f" difference in accuracy (Overfitting): {overfitting:.4f}")

if overfitting > 0.1:
            print("⚠️ : propability of  Overfitting high!")
elif overfitting > 0.05:
            print("⚠️ : propability of  Overfitting medium")
else:
            print("   overfitting good")

actual_classes = sorted(y.unique())

plt.figure(figsize=(20, 10))
plot_tree(dt_classifier,
          feature_names=X.columns,
          class_names=actual_classes,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

plot_tree(dt_classifier,
          feature_names=X.columns,
          class_names=actual_classes,
          filled=True,
          rounded=True,
          fontsize=8,
          ax=axes[0,0])

axes[0,0].set_title('Decision Tree Visualization', fontsize=14, fontweight='bold')

plot_tree(dt_classifier,
          feature_names=X.columns,

          filled=True,
          rounded=True,
          fontsize=8,
          ax=axes[0,1])

axes[0,1].set_title('Decision Tree (Auto Class Names)', fontsize=14, fontweight='bold')


if 'dt_classifier' in locals():
    info_text = f"""
    Model Information:
    Classes: {dt_classifier.classes_}
    Tree Depth: {dt_classifier.get_depth()}
    Number of Leaves: {dt_classifier.get_n_leaves()}

    Features: {list(X.columns)}
    """

    axes[1,0].text(0.1, 0.5, info_text, fontsize=10,
                   verticalalignment='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Model Details', fontsize=14, fontweight='bold')
    axes[1,0].axis('off')

if 'dt_classifier' in locals() and hasattr(dt_classifier, 'feature_importances_'):
    importances = dt_classifier.feature_importances_
    feature_names = X.columns


    indices = np.argsort(importances)[::-1]

    axes[1,1].bar(range(len(importances)), importances[indices])
    axes[1,1].set_title('Feature Importance', fontsize=14, fontweight='bold')
    axes[1,1].set_xticks(range(len(importances)))
    axes[1,1].set_xticklabels([feature_names[i] for i in indices], rotation=45)

plt.tight_layout()
plt.show()

basic_model = DecisionTreeClassifier(random_state=42)
basic_model.fit(X_train, y_train)

interactive_plot = interact(
        build_decision_tree,
        max_depth=IntSlider(
            value=5,
            min=2,
            max=20,
            step=1,
            description='Max Depth:',
            style={'description_width': 'initial'}
        ),
        min_samples_split=IntSlider(
            value=20,
            min=2,
            max=100,
            step=5,
            description='Min Samples Split:',
            style={'description_width': 'initial'}
        ),
        min_samples_leaf=IntSlider(
            value=10,
            min=1,
            max=50,
            step=2,
            description='Min Samples Leaf:',
            style={'description_width': 'initial'}
        ),
        criterion=Dropdown(
            value='gini',
            options=['gini', 'entropy'],
            description='Criterion:',
            style={'description_width': 'initial'}
        ),
        max_features=Dropdown(
            value='sqrt',
            options=['sqrt', 'log2', None],
            description='Max Features:',
            style={'description_width': 'initial'}
        ),
        random_state=IntSlider(
            value=42,
            min=1,
            max=100,
            step=1,
            description='Random State:',
            style={'description_width': 'initial'}
        ),
        show_tree=Checkbox(
            value=True,
            description='Show Tree Visualization'
        ),
        show_metrics=Checkbox(
            value=True,
            description='Show Metrics'
        )
    )

experiments = [
    {
        'name': 'Shallow Tree (depth=3)',
        'params': {'max_depth': 3, 'min_samples_split': 20}
    },
    {
        'name': 'Medium Tree (depth=5)',
        'params': {'max_depth': 5, 'min_samples_split': 15}
    },
    {
        'name': 'Deep Tree (depth=10)',
        'params': {'max_depth': 10, 'min_samples_split': 10}
    },
    {
        'name': 'Entropy Criterion',
        'params': {'max_depth': 5, 'criterion': 'entropy'}
    },
    {
        'name': 'High Min Samples',
        'params': {'max_depth': 5, 'min_samples_split': 50, 'min_samples_leaf': 20}
    }
]

results = []

"""# Task 8: Build a Random Forest Classifier with Interactive Controls"""

ages = np.random.randint(22, 65, n_samples)
experience = np.random.randint(0, np.minimum(40, ages-20), n_samples)
satisfaction = np.random.uniform(1, 10, n_samples)
salary_numeric = np.random.uniform(25000, 200000, n_samples)
work_hours = np.random.uniform(35, 65, n_samples)
projects_count = np.random.poisson(lam=3, size=n_samples)
training_hours = np.random.exponential(scale=20, size=n_samples)
commute_time = np.random.uniform(10, 120, n_samples)

leave_probability = (
    (10 - satisfaction) * 0.08 +
    (work_hours - 45) * 0.015 +
    (commute_time - 30) * 0.005 +
    (100000 - salary_numeric) * 0.000008 +
    np.maximum(0, experience - 15) * 0.02 +
    (ages > 50).astype(int) * 0.1 +
    (projects_count > 5).astype(int) * 0.15
)

leave_probability += np.random.normal(0, 0.1, n_samples)
leave_probability = np.clip(leave_probability, 0.05, 0.95)
target = np.random.binomial(1, leave_probability)

df

df.shape

last_evaluation_counts = Counter(df['last_evaluation'])

for class_label, count in last_evaluation_counts.items():
    percentage = (count / len(df)) * 100
    class_name = "leave" if class_label == 0 else "stay"
    print(f"  {class_name} ({class_label}): {count} ({percentage:.1f}%)")

last_eval_vc = df['last_evaluation'].value_counts().sort_index()
for value, count in last_eval_vc.items():
    percentage = (count / len(df)) * 100
    status = "leave" if value == 0 else "stay"
    print(f"   {status} ({value}): {count} ({percentage:.1f}%)")

def create_labels(counts):
    """Simple way to create labels"""
    label_map = {0: 'بقي (Stayed)', 1: 'ترك (Left)'}
    return [label_map.get(val, f'فئة {val}') for val in counts.index]

last_eval_counts = df['last_evaluation'].value_counts().sort_index()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))


axes[0, 0].bar([str(x) for x in last_eval_counts.index], last_eval_counts.values,
               color=['lightblue', 'lightcoral'][:len(last_eval_counts)])
axes[0, 0].set_title('توزيع آخر تقييم\n(Last Evaluation Distribution)')
axes[0, 0].set_xlabel('Last Evaluation')
axes[0, 0].set_ylabel('Count')

for i, (idx, count) in enumerate(last_eval_counts.items()):
    pct = (count / len(df)) * 100
    axes[0, 0].text(i, count + 10, f'{pct:.1f}%', ha='center')


if len(last_eval_counts) > 0:

    pie_labels = create_labels(last_eval_counts)



    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(last_eval_counts)]

    axes[0, 1].pie(last_eval_counts.values,
                   labels=pie_labels,
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=colors)
    axes[0, 1].set_title('نسبة آخر تقييم\n(Last Evaluation Proportion)')
else:
    axes[0, 1].text(0.5, 0.5, 'لا توجد بيانات\nNo data available',
                    ha='center', va='center')
    axes[0, 1].set_title('No Data')


if 'left' in df.columns:

    crosstab = pd.crosstab(df['last_evaluation'], df['left'])


    crosstab.plot(kind='bar', ax=axes[1, 0],
                  color=['green', 'red'],
                  alpha=0.7)
    axes[1, 0].set_title('العلاقة بين التقييم والترك\n(Evaluation vs Leaving)')
    axes[1, 0].set_xlabel('Last Evaluation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(['Stayed (0)', 'Left (1)'])
    axes[1, 0].tick_params(axis='x', rotation=0)


if 'satisfaction_level' in df.columns:
    axes[1, 1].hist(df['satisfaction_level'], bins=20, alpha=0.7,
                    color='skyblue', edgecolor='black')
    axes[1, 1].set_title('توزيع مستوى الرضا\n(Satisfaction Distribution)')
    axes[1, 1].set_xlabel('Satisfaction Level')
    axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

def build_random_forest(n_estimators, max_depth, min_samples_split):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_
    return rf, None, X_train, X_val, importance

# Call the function
basic_rf, basic_dt, basic_rf_train, basic_rf_val, basic_importance = build_random_forest(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10
)

basic_rf, basic_dt, basic_rf_train, basic_rf_val, basic_importance = build_random_forest(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10
)

interactive_rf = interact(
        build_random_forest,
        n_estimators=IntSlider(
            value=100,
            min=10,
            max=300,
            step=10,
            description='عدد الأشجار:',
            style={'description_width': 'initial'}
        ),
        max_depth=IntSlider(
            value=10,
            min=3,
            max=25,
            step=1,
            description='أقصى عمق:',
            style={'description_width': 'initial'}
        ),
        min_samples_split=IntSlider(
            value=10,
            min=2,
            max=50,
            step=2,
            description='Min Samples Split:',
            style={'description_width': 'initial'}
        ),
        min_samples_leaf=IntSlider(
            value=5,
            min=1,
            max=25,
            step=1,
            description='Min Samples Leaf:',
            style={'description_width': 'initial'}
        ),
        max_features=Dropdown(
            value='sqrt',
            options=['sqrt', 'log2', None, 0.5, 0.8],
            description='Max Features:',
            style={'description_width': 'initial'}
        ),
        bootstrap=Checkbox(
            value=True,
            description='Bootstrap Sampling'
        ),
        random_state=IntSlider(
            value=42,
            min=1,
            max=100,
            step=1,
            description='Random State:',
            style={'description_width': 'initial'}
        ),
        show_tree_sample=Checkbox(
            value=True,
            description='Show Sample Trees'
        ),
        show_comparison=Checkbox(
            value=True,
            description='Show RF vs DT Comparison'
        ),
        show_metrics=Checkbox(
            value=True,
            description='Show Detailed Metrics'
        )
    )

individual_scores = np.mean([tree.predict_proba(basic_rf_val)[:, 1] for tree in basic_rf.estimators_], axis=0)

optimized_acc = accuracy_score(y_val, individual_scores > 0.5)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(individual_scores, bins=8, alpha=0.7, color='forestgreen', edgecolor='black')
plt.axvline(np.mean(individual_scores), color='red', linestyle='--',
           label=f'متوسط الأشجار: {np.mean(individual_scores):.3f}')
plt.axvline(optimized_acc, color='blue', linestyle='-', linewidth=2,
           label=f'Random Forest: {optimized_acc:.3f}')
plt.xlabel('odd accuracy')
plt.ylabel('التكرار')
plt.title('Distribuation of odd accuracy')
plt.legend()


plt.subplot(1, 2, 2)

stability_scores = []
for rs in range(10, 20):
    temp_rf = RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)
    temp_rf.fit(X_train, y_train)
    temp_pred = temp_rf.predict(X_val)
    temp_acc = accuracy_score(y_val, temp_pred)
    stability_scores.append(temp_acc)


dt_stability_scores = []
for rs in range(10, 20):
    temp_dt = DecisionTreeClassifier(max_depth=12, random_state=rs)
    temp_dt.fit(X_train, y_train)
    temp_pred = temp_dt.predict(X_val)
    temp_acc = accuracy_score(y_val, temp_pred)
    dt_stability_scores.append(temp_acc)

plt.boxplot([dt_stability_scores, stability_scores],
           labels=['Decision Tree', 'Random Forest'])
plt.ylabel('Accuracy')
plt.title('  Random States ')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()




