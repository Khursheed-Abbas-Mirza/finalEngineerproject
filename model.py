import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')
# Load dataset
df=pd.read_csv("datasets/obseity.csv")  # Replace with actual dataset
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
X = df[["Age", "Height", "Weight", "Gender", "BMI"]]  # Inputs
y = df["Obesity"]  # Target variable
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy }%")
joblib.dump(model, "obesity.pkl")