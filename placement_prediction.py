import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("placement.csv")

print("Dataset Preview:")
print(data.head())

# Convert Yes/No to numeric
le = LabelEncoder()
data['Placed'] = le.fit_transform(data['Placed'])

print("\nDataset Info:")
print(data.info())

# Features and Target
X = data.drop('Placed', axis=1)
y = data['Placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------- Manual Input Part --------

def get_float_input(msg):
    while True:
        try:
            return float(input(msg))
        except:
            print("❌ Please enter a valid number!")

def get_int_input(msg):
    while True:
        try:
            return int(input(msg))
        except:
            print("❌ Please enter a valid integer!")

print("\n--- Manual Prediction Demo ---")

cgpa = get_float_input("Enter CGPA: ")
tenth = get_float_input("Enter 10th percentage: ")
twelfth = get_float_input("Enter 12th percentage: ")
internships = get_int_input("Enter number of internships: ")
projects = get_int_input("Enter number of projects: ")

input_data = np.array([[cgpa, tenth, twelfth, internships, projects]])
result = model.predict(input_data)

if result[0] == 1:
    print("✅ Student will be PLACED")
else:
    print("❌ Student will NOT be placed")

# -------- Graph at the end --------
plt.figure(figsize=(6,4))
sns.countplot(x='Placed', data=data)
plt.title("Placement Distribution (0 = Not Placed, 1 = Placed)")
plt.show()
