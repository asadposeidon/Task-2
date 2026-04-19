# Fast Credit Card Fraud Detection

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1 Load dataset
train = pd.read_csv("fraudTrain.csv")

print("Dataset Loaded")


# 2 Take small sample for fast training
train = train.sample(n=50000, random_state=42)


# 3 Remove text columns
remove_cols = [
'merchant','category','first','last','street',
'city','state','job','dob','trans_num','trans_date_trans_time'
]

train = train.drop(columns=remove_cols, errors="ignore")


# 4 Convert categorical → numeric
train = pd.get_dummies(train)


# 5 Features and Target
X = train.drop("is_fraud", axis=1)
y = train["is_fraud"]


# 6 Train Test Split
X_train,X_val,y_train,y_val = train_test_split(
X,y,test_size=0.2,random_state=42
)


# 7 Decision Tree (very fast)
dt = DecisionTreeClassifier(max_depth=8)

dt.fit(X_train,y_train)

pred_dt = dt.predict(X_val)

print("Decision Tree Accuracy:",accuracy_score(y_val,pred_dt))


# 8 Random Forest (optimized)
rf = RandomForestClassifier(
n_estimators=40,
max_depth=10,
n_jobs=-1
)

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_val)

print("Random Forest Accuracy:",accuracy_score(y_val,pred_rf))



plt.show()