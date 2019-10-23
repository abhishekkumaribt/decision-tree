# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the train data stored in path variable
train = pd.read_csv(path)

# Load the test data stored in path1 variable
test = pd.read_csv(path1)

# necessary to remove rows with incorrect labels in test dataset
test = test.dropna(subset=["Target"])

# encode target variable as integer
le = LabelEncoder()
le.fit(train.Target)
train.Target = le.transform(train.Target)
test.Target = test.Target.apply(lambda x: x[:-1])
test.Target = le.transform(test.Target)

# Plot the distribution of each feature
cols_num = train.select_dtypes(include=np.number).columns[:-1]
fig, ax = plt.subplots(3, 2, figsize=[20,20])
for i in range(3):
    for j in range(2):
        train[cols_num[i*2+j]].plot(kind="hist", ax=ax[i][j])
        ax[i][j].set_xlabel(cols_num[i*2+j])

cols_cat = train.select_dtypes(include='object').columns
fig, ax = plt.subplots(4, 2, figsize=[20,40])
for i in range(4):
    for j in range(2):
        train[cols_cat[i*2+j]].value_counts().plot(kind="bar", ax=ax[i][j])
        ax[i][j].set_xlabel(cols_cat[i*2+j])

# convert the data type of Age column in the test data to int type
test.Age = test["Age"].astype("int")

# cast all float features to int type to keep types consistent between our train and test data

 
# choose categorical and continuous features from data and print them


# fill missing data for catgorical columns
for col in cols_cat:
    train[col] = train[col].fillna(train[col].mode()[0])

for col in cols_cat:
    test[col] = test[col].fillna(test[col].mode()[0])

# fill missing data for numerical columns   
for col in cols_num:
    train[col] = train[col].fillna(train[col].mean())

for col in cols_cat:
    test[col] = test[col].fillna(test[col].mode()[0])

# Dummy code Categoricol features
for col in cols_cat:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Check for Column which is not present in test data


# New Zero valued feature in test data for Holand


# Split train and test data into X_train ,y_train,X_test and y_test data
X_train = train.drop(columns="Target")
y_train = train.Target
X_test = test.drop(columns="Target")
y_test = test.Target

# train a decision tree model then predict our test data and compute the accuracy
dec_tree = DecisionTreeClassifier(max_depth=3, random_state=17)
dec_tree.fit(X_train, y_train)
dec_score = dec_tree.score(X_test, y_test)
print(dec_score)

# Decision tree with parameter tuning
tree_params = {'max_depth' : range(2,11)}
dt = DecisionTreeClassifier(random_state=17)
grid = GridSearchCV(dt, tree_params)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_params_)

# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_


#train a decision tree model with best parameter then predict our test data and compute the accuracy




