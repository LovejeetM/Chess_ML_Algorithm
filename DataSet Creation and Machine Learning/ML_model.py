import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

#For creating machine learning model from dataset

sc = StandardScaler()
le= LabelEncoder()
white = LogisticRegression()
black = SVC()

dfw= pd.read_csv('dataset/white/white.csv')
dfb= pd.read_csv('dataset/black/black.csv')

x_w = dfw.iloc[:, :-1]
y_w = dfw.iloc[:, -1] 


x_train_w, x_test_w, y_train_w, y_test_w = train_test_split(x_w, y_w, test_size=0.2, random_state=42)

x_train_w= sc.fit_transform(x_train_w)
x_test_w= sc.transform(x_test_w)


white.fit(x_train_w, y_train_w)

y_pred_w = white.predict(x_test_w)
print("Accuracy white:", accuracy_score(y_test_w, y_pred_w))
print("Precision white:", precision_score(y_test_w, y_pred_w, average='weighted'))
print("Recall white:", recall_score(y_test_w, y_pred_w, average='weighted'))
print("F1 Score white:", f1_score(y_test_w, y_pred_w, average='weighted'))

print("\nClassification Report for White:")
print(classification_report(y_test_w, y_pred_w))


cm = confusion_matrix(y_test_w, y_pred_w, labels=white.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=white.classes_, yticklabels=white.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

x_b = dfb.iloc[:, :-1]
y_b = dfb.iloc[:, -1] 


x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_b, y_b, test_size=0.2, random_state=42)

x_train_b= sc.fit_transform(x_train_b)
x_test_b= sc.transform(x_test_b)

black.fit(x_train_b, y_train_b)

y_pred_b = black.predict(x_test_b)
print("Accuracy black:", accuracy_score(y_test_b, y_pred_b))
print("Precision black:", precision_score(y_test_b, y_pred_b, average='weighted'))
print("Recall black:", recall_score(y_test_b, y_pred_b, average='weighted'))
print("F1 Score black:", f1_score(y_test_b, y_pred_b, average='weighted'))


print("\nClassification Report for Black:")
print(classification_report(y_test_b, y_pred_b))


cm = confusion_matrix(y_test_b, y_pred_b, labels=black.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=black.classes_, yticklabels=black.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

joblib.dump(white, 'white.pkl')
joblib.dump(black, 'black.pkl')
joblib.dump(sc, 'scaler.pkl')
