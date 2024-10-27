import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Basic statistics
print(data.describe())

# Check data types
print(data.info())

##Visualize the Data---
import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution (0 = Genuine, 1 = Fraud)
# Class distribution (0 = Genuine, 1 = Fraud)
sns.countplot(data=data, x='Class', hue='Class', palette='coolwarm', dodge=False, legend=False)
plt.title('Class Distribution')
plt.show()


### Handle Class Imbalance-----

# Separate the data into fraud and non-fraud transactions
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(len(fraud))

# Combine fraud and non-fraud transactions
balanced_data = pd.concat([fraud, non_fraud])

# Visualize the new class distribution
sns.countplot(x='Class', data=balanced_data)
plt.title('Balanced Class Distribution')
plt.show()

##Preprocess the Data-----

from sklearn.preprocessing import StandardScaler

# Features and target variable
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

##Split the Data into Train and Test Sets----

from sklearn.model_selection import train_test_split

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

##Train the Model(logical regression)

from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

##Evaluate the Model

from sklearn.metrics import classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

##Improve Model Performance(Random Forests)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
