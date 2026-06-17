import pandas as pd

# Load dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\Downloads\DAY 01\iris.csv')

# Show first 5 rows
print(dataset.head())

# Shape of dataset
print("Dataset Shape:", dataset.shape)


####################################################################################################


# Features (first 4 columns)
data = dataset.iloc[:, 0:4]

# Target (last column)
target = dataset.iloc[:, 4]

print("Data Shape:", data.shape)
print("Target Shape:", target.shape)


####################################################################################################



# Split dataset
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42
)

print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("Train Target Shape:", train_target.shape)
print("Test Target Shape:", test_target.shape)


####################################################################################################


# Import KNN
from sklearn.neighbors import KNeighborsClassifier

# Create model
model = KNeighborsClassifier(n_neighbors=3)#onimathnaa 3 dena eka his thiyanna puluwan

# Train model
model.fit(train_data, train_target)

# Predict
predicted_target = model.predict(test_data)

print("\nActual Target:")
print(test_target.values)

print("\nPredicted Target:")
print(predicted_target)


####################################################################################################


# Accuracy
from sklearn.metrics import accuracy_score

acc = accuracy_score(test_target, predicted_target)

print("\nAccuracy:", acc)


####################################################################################################



# Save model
import joblib

joblib.dump(model, 'KNN_iris_model.sav')

print("\nModel saved successfully!")