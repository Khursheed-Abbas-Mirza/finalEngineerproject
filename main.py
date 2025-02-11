import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')
# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with actual dataset

# Data preprocessing
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Example of categorical encoding

joblib.dump(le, 'scaler.pkl')
y = df['ObesityCategory']  # Target variable (Obesity level)
X = df.drop(columns=['ObesityCategory','PhysicalActivityLevel'])  # Features
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')   # Save the scaler for future use
# Train ML model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model,'obseity.pkl')
# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Example: Predict obesity level for new input
def predict_obesity(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]

# Example input (replace with real values)
new_input = [1, 25, 175, 80,26.1]  # Gender, Age, Height (cm), Weight (kg),BMI
new_input=[1,20,175,64,21.2]
with open('dataset.csv','r') as f:
    data=f.readlines()
for i in data[1:5]:
    y=i.split(',')
    y[1]=1 if y[1] =='Male' else 0
    newinput=[y[1],y[0],y[2],y[3],y[4]]
    print("Predicted obseity level",predict_obesity(newinput))
print("Predicted Obesity Level:", predict_obesity(new_input))