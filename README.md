# Wine Quality Prediction

## Project Overview

This project aims to predict the quality of red wine using various machine learning models. The workflow includes data preprocessing, feature analysis, model building, and hyperparameter tuning to achieve accurate predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Analysis](#feature-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

The quality of wine is influenced by various chemical properties. This project leverages machine learning techniques to predict the quality of red wine based on these properties.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Cleaning**:
   - Verified that the dataset has no missing values.

3. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Feature Analysis

1. **Correlation Analysis**:
   - Visualized and identified significant correlations between features using heatmaps.

2. **Visualization**:
   - Used bar plots, scatter plots, and histograms to understand feature distributions and relationships.

## Modeling

1. **Logistic Regression**:
   - Built a logistic regression model to classify wine quality.

2. **Random Forest Classifier**:
   - Developed a Random Forest model and optimized it using GridSearchCV for better performance.

## Results

- **Logistic Regression**:
  - Training Accuracy: 0.87568412
  - Testing Accuracy: 0.89375

- **Random Forest Classifier**:
  - Training Accuracy: 1.0
  - Testing Accuracy: 0.93125
  - Best Estimator: (RandomForestClassifier(max_depth=10, n_estimators=24), 0.9014756944444444)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/wine-quality.git
   ```

2. Navigate to the project directory:
   ```bash
   cd wine-quality-prediction
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict wine quality based on new data.

## Conclusion

This project demonstrates the use of machine learning models to predict the quality of red wine. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors influencing wine quality.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("D:\\Courses language programming\\Machine Learning\\Folder Machine Learning\\Red_Wine_Quality\\winequality-red.csv")

# Data information
data.info()
data.describe()

# Visualizations
data["quality"].value_counts().plot(kind="bar", color="green")
plt.title("Quality of Wine Vs Count Value")
plt.xlabel("Quality Value")
plt.ylabel("Count Of Value")
plt.grid()
plt.show()

data.hist(figsize=(20, 10))

plt.figure(figsize=(20, 10))
x = plt.scatter(x=data["quality"], y=data["volatile acidity"], c=data["alcohol"], cmap=plt.get_cmap("jet"), alpha=0.7, edgecolors="black", linewidths=1, s=70)
plt.colorbar(x)
plt.show()

sns.barplot(x=data["quality"], y=data["citric acid"])
sns.barplot(x=data["quality"], y=data["total sulfur dioxide"])

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), cbar=True, annot=True, fmt=".1f", cmap="Blues", square=True)

# Data splitting
X = data.drop(columns="quality", axis=1)
Y = data["quality"].apply(lambda y_value: 1 if y_value >= 7 else 0)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=2, stratify=Y)

# Logistic Regression
model_1 = LogisticRegression(max_iter=1000)
model_1.fit(x_train, y_train)
print(f"The prediction Value Training Data is  ==> {model_1.score(x_train, y_train)}")
print(f"The prediction Value Testing Data is  ==> {model_1.score(x_test, y_test)}")

# Random Forest Classifier
model_2 = RandomForestClassifier()
model_2.fit(x_train, y_train)
print(f"The prediction Value Training Data is  ==> {model_2.score(x_train, y_train)}")
print(f"The prediction Value Testing Data is  ==> {model_2.score(x_test, y_test)}")

# Hyperparameter tuning
param = {"n_estimators": np.arange(20, 26, 1), "max_depth": np.arange(9, 11, 1), "min_samples_split": [2, 3, 4]}
new_model = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param, cv=20, n_jobs=-1, scoring="accuracy", refit=0.4)
new_model.fit(x_train, y_train)
print("Best Estimator:", new_model.best_estimator_)
print("Best Score:", new_model.best_score_)

# Prediction system
Data_Prediction = np.asarray(list(map(float, input().split(",")))).reshape(1, -1)
Predict = model_2.predict(Data_Prediction)
print("-" * 30)
if Predict[0] == 1:
    print("Quality Wine is Good")
else:
    print("Quality Wine is Bad")
print("-" * 30)
```
