import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# fetching the diabetes dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")

# loading the dataset
diabetes_dataset = pd.read_csv(DATA_PATH)

# Exploratory Data Analysis (EDA)
print("Shape of the dataset: " + str(diabetes_dataset.shape))

print("Statistical summary of the dataset: ")
print(diabetes_dataset.describe())

print("Columns in the dataset: " + str(diabetes_dataset.columns))

# Mean values of each feature grouped by the Outcome variable
print("Mean values of each feature grouped by Outcome: ")
print(diabetes_dataset.groupby('Outcome').mean())

# Correlation between the features
print("Correlation of the features: ")
print(diabetes_dataset.corr())

print("Distribution of the Outcome variable: ")
print(diabetes_dataset['Outcome'].value_counts())
# 0 --> Non-Diabetic
# 1 --> Diabetic

# plotting the correlation heatmap
plt.figure(figsize=(10, 8))
correlation = diabetes_dataset.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Diabetes Dataset Features')
plt.savefig(os.path.join(BASE_DIR, "dataset/graph-representation-of-dataset", "correlation_heatmap.png"))
plt.show()

# Visualizing the distribution of the Outcome variable
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Distribution of Outcome Variable')
plt.savefig(os.path.join(BASE_DIR, "dataset/graph-representation-of-dataset", "outcome_distribution.png"))
plt.show()

# Checking for missing values
print("Missing values in each column: ")
print(diabetes_dataset.isnull().sum())

# Checking for zero values in the dataset
print("Count of zero values in each column: ")
print((diabetes_dataset == 0).sum())

# Function to preprocess the data by replacing zero values with the median of the respective columns
def preprocess_data(df):
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df[col] = df[col].replace(0, df[col].median())
    return df

# Preprocess the data
preprocess_data(diabetes_dataset)
print((diabetes_dataset == 0).sum())

print("Data preprocessing completed.")
# Note: The diabetes dataset does not contain any missing values as per the above check.