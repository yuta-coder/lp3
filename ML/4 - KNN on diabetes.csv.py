# practical4_diabetes_knn.py
# Requirements: pandas, numpy, sklearn, matplotlib, seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

df = pd.read_csv("diabetes.csv")   # Kaggle file

# Replace zeros in invalid columns with median
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols:
    df[c] = df[c].replace(0, np.nan)
    df[c].fillna(df[c].median(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Split
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)
pred = knn.predict(Xte)

# Metrics
cm = confusion_matrix(yte, pred)
acc = accuracy_score(yte, pred)
err = 1 - acc
prec = precision_score(yte, pred)
rec = recall_score(yte, pred)
print("Accuracy:", round(acc,4))
print("Error rate:", round(err,4))
print("Precision:", round(prec,4))
print("Recall:", round(rec,4))
print("\nClassification report:\n", classification_report(yte, pred, digits=4))

# Visuals: confusion matrix heatmap and a bar of metrics
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (KNN)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(5,3))
sns.barplot(x=["Accuracy","Precision","Recall","Error"], y=[acc, prec, rec, err])
plt.ylim(0,1)
plt.title("KNN metrics")
plt.show()
