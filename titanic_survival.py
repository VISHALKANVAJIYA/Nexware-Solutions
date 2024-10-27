import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
try:
    data = pd.read_csv('tested.csv')
except FileNotFoundError:
    print("Error: File not found. Ensure the file path is correct.")
    exit()

# Explore the data
print(data.head(), "\n")
print(data.isnull().sum(), "\n")
print(data.describe(), "\n")
print(data.info(), "\n")

# Visualize class distribution
sns.countplot(x='Survived', data=data, palette='coolwarm')
plt.title('Survival Distribution')
plt.show()

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data.drop('Cabin', axis=1, inplace=True)
data.dropna(subset=['Embarked'], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns
data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
