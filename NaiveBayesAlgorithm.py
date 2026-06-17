import pandas as pd

# Load dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\Downloads\DAY 01\iris.csv')

print(dataset.head())
print(dataset.shape)


####################################################################################################



# Features
data = dataset.iloc[:, 0:4]

# Target
target = dataset.iloc[:, 4]


####################################################################################################



# Split dataset
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42
)


####################################################################################################



# Import Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Create model
model = GaussianNB()

# Train model
model.fit(train_data, train_target)

# Predict
predicted_target = model.predict(test_data)

print("Actual Target:")
print(test_target.values)

print("Predicted Target:")
print(predicted_target)


####################################################################################################


# Accuracy
from sklearn.metrics import accuracy_score

acc = accuracy_score(test_target, predicted_target)

print("Accuracy:", acc)


####################################################################################################


# Save model
import joblib

joblib.dump(model, 'NB_iris_model.sav')

print("Model saved successfully!")