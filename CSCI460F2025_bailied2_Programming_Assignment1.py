######################################################
#  Student Name: Devin Bailie
#  Student ID: bailied2 / W30644462
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: 10/14/2025
#  GitHub Link: https://github.com/bailied2/csci460-naive-bayes
######################################################
import pandas as pd 
import sklearn.naive_bayes

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
  "illiterate":0,
  "high.school":1,
  "basic.4y":2,
  "basic.6y":3,
  "basic.9y":4,
  "university.degree":5,
  "professional.course":6,
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
  "nonexistent":2,
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


print(df.to_string(max_rows=10))
