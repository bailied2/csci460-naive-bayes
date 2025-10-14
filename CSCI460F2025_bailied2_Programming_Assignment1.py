######################################################
#  Student Name: Devin Bailie
#  Student ID: bailied2 / W30644462
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: 10/14/2025
#  GitHub Link: https://github.com/bailied2/csci460-naive-bayes
######################################################

# Import pandas for dataframe
import pandas as pd 
# Import model selection function
from sklearn.model_selection import train_test_split
# Import classification algorithm
from sklearn.naive_bayes import GaussianNB
# Import performance metric libraries
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  ConfusionMatrixDisplay,
  f1_score,
  classification_report,
)

# Import data from bank-full.csv
df = pd.read_csv('/home/acc.bailied2/csci460/csci460-naive-bayes/bank-full.csv')

# Categorical variable maps
jobMap = {
  "admin.":0, 
  "blue-collar":1, 
  "entrepreneur":2, 
  "housemaid":3, 
  "management":4, 
  "retired":5,
  "self-employed":6,
  "services":7,
  "student":8,
  "technician":9,
  "unemployed":10,
  }
maritalMap = {
  "single":0,
  "married":1,
  "divorced":2,
}
educationMap = {
  "primary":0,
  "secondary":1,
  "tertiary":2,
}
contactMap = {
  "cellular":0,
  "telephone":1,
}
monthMap = {
  "jan":0,
  "feb":1,
  "mar":2,
  "apr":3,
  "may":4,
  "jun":5,
  "jul":6,
  "aug":7,
  "sep":8,
  "oct":9,
  "nov":10,
  "dec":11,
}
poutcomeMap = {
  "failure":0,
  "success":1,
  "other":2,
}
binaryMap = {
  "no":0,
  "yes":1,
}

# Apply maps to dataframe
df["job"] = df["job"].map(jobMap)
df["marital"] = df["marital"].map(maritalMap)
df["education"] = df["education"].map(educationMap)
df["default"] = df["default"].map(binaryMap)
df["housing"] = df["housing"].map(binaryMap)
df["loan"] = df["loan"].map(binaryMap)
df["contact"] = df["contact"].map(contactMap)
df["month"] = df["month"].map(monthMap)
df["poutcome"] = df["poutcome"].map(poutcomeMap)
df["y"] = df["y"].map(binaryMap)

# Remove rows with missing cells
df.dropna(inplace = True)

# Some columns automatically get set as floats.
# Let us set them all to integers.
df = df.astype("int64")

# Set X and y variables
X = df.drop("y", axis=1)
y = df["y"]

# TASK 2: 70% training, 30% testing
for i in range(10):
  # Create training and testing data sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

  # Create Gaussian Naive Bayes classifier instance
  gnb = GaussianNB()

  # Train (or fit) classifier to the training data
  gnb.fit(X_train, y_train)

  # Use the trained model to generate predictions for the test data
  y_pred = gnb.predict(X_test)

  # measure the performance
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average="weighted")
  print("\nITERATION", i+1, "\n---------")
  print("The accuracy of my Naive Bayes Model is:", accuracy)
  print("The F1 Score of my Naive Bayes Model is:", f1)