import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay

# Create images directory if not exists
os.makedirs("images", exist_ok=True)

# ==================== Loading Data ====================
df = pd.read_csv('employee_data.csv')
print("Dataset Shape:", df.shape)

# ==================== Missing Data Handling ====================
missing_percent = (df.isnull().sum() / len(df)) * 100
df_processed = df.copy()

# Drop columns with >50% missing data
columns_to_drop = missing_percent[missing_percent > 50].index.tolist()
df_processed = df_processed.drop(columns=columns_to_drop)

# Fill numeric columns with median
num_columns = df_processed.select_dtypes(include=[np.number]).columns
for col in num_columns:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_columns = df_processed.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# ==================== Encoding ====================
for col in categorical_columns:
    if df_processed[col].nunique() == 2:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

multi_cat_columns = [col for col in categorical_columns if 2 < df_processed[col].nunique() <= 10]
df_processed = pd.get_dummies(df_processed, columns=multi_cat_columns, prefix=multi_cat_columns)

# ==================== Scaling ====================
numeric_features = df_processed.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])

# ==================== Target & Features ====================
target_column = 'quit'
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

# ==================== Train-Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==================== Handle Imbalance ====================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ==================== Decision Tree ====================
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
dt_classifier.fit(X_train, y_train)
y_test_pred_dt = dt_classifier.predict(X_test)

# ==================== Random Forest ====================
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42)
rf_classifier.fit(X_train, y_train)
y_test_pred_rf = rf_classifier.predict(X_test)

# ==================== Results ====================
results_df = pd.DataFrame([
    {"Model": "Decision Tree", "Accuracy": accuracy_score(y_test, y_test_pred_dt), "Precision": precision_score(y_test, y_test_pred_dt, average='weighted'), "Recall": recall_score(y_test, y_test_pred_dt, average='weighted'), "F1-Score": f1_score(y_test, y_test_pred_dt, average='weighted')},
    {"Model": "Random Forest", "Accuracy": accuracy_score(y_test, y_test_pred_rf), "Precision": precision_score(y_test, y_test_pred_rf, average='weighted'), "Recall": recall_score(y_test, y_test_pred_rf, average='weighted'), "F1-Score": f1_score(y_test, y_test_pred_rf, average='weighted')}
])
print(results_df)

# ==================== Save Plots ====================

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_processed.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.close()

# Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['blue', 'green']

for ax, metric in zip(axes.flatten(), metrics):
    ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(f'Model {metric} Comparison')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=300)
plt.close()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred_dt)).plot(ax=axes[0])
axes[0].set_title('Decision Tree')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred_rf)).plot(ax=axes[1])
axes[1].set_title('Random Forest')
plt.tight_layout()
plt.savefig("images/confusion_matrices.png", dpi=300)
plt.close()

print("All plots saved in 'images/' folder.")
