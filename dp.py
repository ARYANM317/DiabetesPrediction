#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#database loading and pre-processing
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import (classification_report, accuracy_score, 
                             precision_recall_fscore_support, confusion_matrix, 
                             precision_score, recall_score, roc_auc_score, f1_score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r'/diabetes_prediction_dataset.csv')

df.head()

df.describe()

df.info()

# Separate features X and target y  
target = 'diabetes'
X = df.drop(target, axis=1)
y = df[[target]]

from sklearn.compose import ColumnTransformer

# Initialize encoders and scaler
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()
le = LabelEncoder()

# Convert column names to lowercase and strip whitespace
X.columns = X.columns.str.strip().str.lower()

# Define categorical and numeric variables
categorical_variables = ['gender']
# Updated numeric variable names to match the lowercase and stripped column names
numeric_variables = ['age', 'hypertension', 'heart_disease', 'bmi', 'hba1c_level', 'blood_glucose_level']  

# Define column transformations
transformer = ColumnTransformer(
    transformers=[('ohe', encoder, categorical_variables),
                  ('scaler', scaler, numeric_variables)
                 ],
    remainder="passthrough",
)
transformer.set_output(transform="pandas")

# Fit and transform feature data
X = transformer.fit_transform(X)
# Fit and transform target labels
le.fit(y.values.ravel())
y[target] = le.fit_transform(y[target])

# Split the data into training and validational samples
rs=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)
print('\nAfter Pre-processing:')
print('Size of train dataset: ' + str(y_train.shape[0])) 
print('Size of test dataset: ' + str(y_test.shape[0])) 

#DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=rs)

# Train the model
DTC.fit(X_train, y_train)
y_pred_test = DTC.predict(X_test)

DTC_original = evaluate(y_test, y_pred_test)

DTC_original

#RandomForestClassifier
RFC = RandomForestClassifier(random_state=rs)
RFC.fit(X_train, y_train)

y_pred_test = RFC.predict(X_test)
RFC_original = evaluate(y_test, y_pred_test)
RFC_original
