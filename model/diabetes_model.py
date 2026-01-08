import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Function to preprocess the data by replacing zero values with the median of the respective columns
def preprocess_data(df):
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df[col] = df[col].replace(0, df[col].median())
    return df

# loading the diabetes dataset to a pandas DataFrame
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")

# loading the dataset
diabetes_dataset = pd.read_csv(DATA_PATH)
# preprocessing the data

diabetes_dataset = preprocess_data(diabetes_dataset)

# print the first 10 rows of the dataset
print(diabetes_dataset.head(10))

# separating the data and labels
X = diabetes_dataset.drop(columns = ['Pregnancies','Outcome'], axis=1)
Y = diabetes_dataset['Outcome']

# print x and y to verify the separation
print('X -> ',X)
print('Y -> ',Y)

# data standardization standardizes the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# assigning the standardized data to X and Y variables again due to transformation
X = standardized_data
Y = diabetes_dataset['Outcome']

# # print the final X and Y
print(X)
print(Y)

# splitting the data into training data and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# print the shapes of the splitted data
print("X shape: ", X.shape)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

# creating a SVM model for training the data
classifier = svm.SVC(kernel="linear", random_state=2, probability=True)

# training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Model Evaluation :-

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

# saving the trained model and scaler using joblib in the artifacts directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")

# create the artifacts directory if it doesn't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# save the trained model and scaler model
joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, "diabetes.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

# confirmation message
print("Model and Scaler saved successfully.")