# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix

df = pd.read_csv("path.csv")

# Shape of the dataset
df.shape

# Columns in the dataset
df.columns

# Head of the dataset
df.head(n=20)

# Target variable values
df["Status"].unique()

# Check null values
df.isnull().sum()

# Treat null values
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage

for column in df.columns:
    if missing_percentage[column] > 70:
        df.drop(column, axis=1, inplace= True)

mode = df['CWE'].mode()[0]
df['CWE'].fillna(mode, inplace=True)

null_counts = df.isnull().sum()
null_counts

# A boxplot for Status with individual data points of Line
# Set figure size

plt.figure(figsize=(10, 8))

sns.boxplot(x='Status', y='Line', data=df, showfliers=True, width=0.8, palette="colorblind")
# sns.stripplot(x='Status', y='Line', data=df, color='black', jitter=0.4, alpha=0.2)

# labels and title
plt.xlabel('Status', fontsize=14)
plt.ylabel('Line', fontsize=14)
plt.title('Box Plot for Status with Individual Data Points of Line', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# A bar graph for frquency distribution
# Set figure size

plt.figure(figsize=(8, 6))

sns.histplot(data=df, x='Status', hue='Severity', multiple='stack', shrink=0.8, palette='colorblind')

# labels and title
plt.xlabel('Status', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Stacked Bar Chart of Frequency Distribution', fontsize=16)
plt.legend(title='Severity')

plt.show()

# Correlation Matrix

from scipy.stats import chi2_contingency

# A DataFrame with categorical data
columns_for_correlation = ['*']
data = df[columns_for_correlation]

# A function to calculate Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# An empty correlation matrix
num_features = len(data.columns)
correlation_matrix = np.zeros((num_features, num_features))

# Calculate Cramer's V for each pair of categorical variables
for i in range(num_features):
    for j in range(num_features):
        if i == j:
            correlation_matrix[i, j] = 1.0
        else:
            correlation_matrix[i, j] = cramers_v(data.iloc[:, i], data.iloc[:, j])

# The correlation matrix
correlation_df = pd.DataFrame(correlation_matrix, columns=data.columns, index=data.columns)

correlation_matrix = correlation_df
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()

# Drop unnecessary columns

for x in ['ID','Tool','Path']:
    df.drop(x, axis=1, inplace= True)

# Input variables for modelling

ip_variables = ['*']

# Transform required columns

X = df[ip_variables]
y = df['Status']

categorical_cols = ['*']
numerical_cols = ['*']

# The transformer
categorical_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

# Create a pipeline
from sklearn.pipeline import Pipeline

preprocessor = Pipeline(
    steps=[
        ('categorical', categorical_transformer)
    ]
)

# Train-test split

X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# A Logistic Regression classifier

lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_classifier.fit(X_train, y_train)
y_lr_pred = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

# The logistic Regression confusion matrix

lg_confusion = confusion_matrix(y_test, y_lr_pred)

plot_confusion_matrix(lr_classifier, X_test, y_test)
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Parameters for a Random Forest Classifier

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=1,  # Adjust the number of iterations
    cv=5
)

random_search.fit(X_train, y_train)

# The best parameters
best_params = random_search.best_params_

# A Random Forest Classifier

best_rf_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf']
)

best_rf_model.fit(X_train, y_train)

y_rf_pred = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# The Random Forest confusion matrix

rf_confusion = confusion_matrix(y_test, y_rf_pred)

plot_confusion_matrix(best_rf_model, X_test, y_test)
plt.title('Random Forest Confusion Matrix')
plt.show()

# An Support Vector Machine Classifier

svm_classifier = SVC(kernel='poly', C=1.0) 
svm_classifier.fit(X_train, y_train)

y_svm_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_svm_pred)
print('Support Vector Machine Accuracy: ', accuracy)

# The Support Vector Machine confusion matrix

SVM_confusion = confusion_matrix(y_test, y_svm_pred)

plot_confusion_matrix(svm_classifier, X_test, y_test)
plt.title('Support Vectyor Machine Confusion Matrix')
plt.show()

# The K-Nearest Neighbor Classifier

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

y_knn_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_knn_pred)
print('The K-Nearest Neighbor Accuracy: ', accuracy)

# The K-Nearest Neighbor confusion matrix
Knn_confusion = confusion_matrix(y_test, y_knn_pred)

plot_confusion_matrix(knn_classifier, X_test, y_test)
plt.title('Knn Confusion Matrix')
plt.show()

