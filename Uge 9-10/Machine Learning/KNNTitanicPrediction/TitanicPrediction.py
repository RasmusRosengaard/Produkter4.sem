import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Select relevant columns
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numerical values
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # Male = 0, Female = 1
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])  # C, Q, S -> 0, 1, 2

# Define features (X) and target (y)
X = df.drop(columns=['Survived'])  # Features
y = df['Survived']  # Target variable (Survived or Not)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5) # k = how many neighbors to consider
knn.fit(X_train, y_train)

# Test the model
y_pred = knn.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---- User Input for Prediction ----
print("\n🔮 Predict a New Passenger's Survival 🔮")

# Get user input
pclass = int(input("Enter Passenger Class (1 = First, 2 = Second, 3 = Third): "))
sex = input("Enter Sex (male/female): ")
age = float(input("Enter Age: "))
sibsp = int(input("Enter Number of Siblings/Spouses aboard: "))
parch = int(input("Enter Number of Parents/Children aboard: "))
fare = float(input("Enter Fare Price: "))
embarked = input("Enter Embarked Port (C = Cherbourg, Q = Queenstown, S = Southampton): ")

# Convert categorical input to numeric
sex = 1 if sex.lower() == 'female' else 0  # Female = 1, Male = 0
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_dict.get(embarked.upper(), 2)  # Default to 'S' if input is invalid

# Prepare the new passenger's data
new_passenger = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
new_passenger = scaler.transform(new_passenger)  # Scale input data

# Predict survival
prediction = knn.predict(new_passenger)
print("\nPrediction:")
print("✅ Survived!" if prediction[0] == 1 else "❌ Did NOT Survive")
