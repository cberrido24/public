#-----------------------------------------------------------------------------#
# Linear regression - based on kaggle competition
# https://www.kaggle.com/competitions/playground-series-s3e16/overview
#
#
# The goal of this competition is to use regression to predict the age of
# crabs given physical attributes.
# First, we will try to understand the data and visualise the relationships
# between the different variables
# After that, we will try to create a model to predict the age of
# the observations on the test dataset
#-----------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------------#
# Importing data
#-----------------------------------------------------------------------------#
path = "D:/projects/linear_reg_crabs/"
inpath = str(path+"inputs/")
outpath = str(path+"outputs/")

train = pd.read_csv(str(inpath+"train.csv"))
test = pd.read_csv(str(inpath+"test.csv"))

#-----------------------------------------------------------------------------#
# Data exploration and visualizations
#-----------------------------------------------------------------------------#

# Distribution of Age
plt.figure(figsize=(10, 6))
plt.hist(train['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Here, we see that the number of crabs is centered around 10 years of age

# Sex Distribution
plt.figure(figsize=(10, 6))
train['Sex'].value_counts().plot(kind='bar', color=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Distribution of Abalones by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# We notice that there are slightly more males than females and infants in our dataset

# Length vs. Weight
plt.figure(figsize=(10, 6))
plt.scatter(train['Length'], train['Weight'], alpha=0.5, color='purple')
plt.title('Length vs. Weight')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.grid(True)
plt.show()

# Here, we notice that there is a positive relationship between the length and the weight of the crabs,
# where animals with greater lengths tend to also be heavier

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# This correlation matrix helps identify which numerical variables are related and to what extent.
# Strong correlations are highlighted between variables like Length, Diameter, and Weight
# We see that as one of these measurements increases, the others tend to increase as well.


#-----------------------------------------------------------------------------#
# Feature selection and preparing data
#-----------------------------------------------------------------------------#
train.columns
test.columns
dict = {"Shucked Weight" : "shucked_weight",
        "Viscera Weight" : "viscera_weight",
        "Shell Weight" : "shell_weight"
        }

train.rename(dict, axis=1, inplace=True)
test.rename(dict, axis=1, inplace=True)

features = [#"Sex",
            "Length",
            "Diameter",
            "Height",
            "Weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight"
            ]

variable_to_predict = np.array(train["Age"])
variables_used = np.array(train[features])
variables_used_test_set = np.array(test[features])

def rename_sex(dataset):
    dataset["sex_I"] = np.where(dataset["Sex"]=="I", 1, 0)
    dataset["sex_M"] = np.where(dataset["Sex"]=="M", 1, 0)
    dataset["sex_F"] = np.where(dataset["Sex"]=="F", 1, 0)

rename_sex(train)
rename_sex(test)

# Normalise values
from sklearn import preprocessing
variables_used = preprocessing.StandardScaler().fit(variables_used).transform(variables_used)

# Split train and test datasets
from sklearn.model_selection import train_test_split
train_features, test_features, train_age, test_age = train_test_split(variables_used, variable_to_predict, test_size=0.2, random_state=123)

print ('Train set:', train_features.shape,  train_age.shape)
print ('Test set:', test_features.shape,  test_age.shape)

#-----------------------------------------------------------------------------#
# Modelling
#-----------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(train_features, train_age)

test_modelled_prediction = np.round(LR.predict(test_features), 0)
print(test_modelled_prediction)

#-----------------------------------------------------------------------------#
# Evaluating model and plots
#-----------------------------------------------------------------------------#
# Evaluating model
MSE = np.mean((test_modelled_prediction - test_age) ** 2)
print("Mean Squared error(MSE): %.2f"
      % MSE)

# Plots
plt.scatter(train['viscera_weight'], train['Age'], color='blue')
plt.xlabel("Viscera weight")
plt.ylabel("Age")
plt.show()

#-----------------------------------------------------------------------------#
# Predicting
#-----------------------------------------------------------------------------#
# Normalise values
variables_used_test_set = preprocessing.StandardScaler().fit(variables_used_test_set).transform(variables_used_test_set)
modelled_predictions_test_set = np.round(LR.predict(variables_used_test_set), 0)

#-----------------------------------------------------------------------------#
# Exporting results
#-----------------------------------------------------------------------------#
predictions_to_submit = pd.concat([test["id"], pd.DataFrame(modelled_predictions_test_set)], axis=1)
predictions_to_submit = predictions_to_submit.rename({0:'Age'}, axis=1)
predictions_to_submit.to_csv(str(outpath+"submission.csv"), index=False)


