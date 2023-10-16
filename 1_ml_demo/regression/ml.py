import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os


method='svr' # 'dcst'

# Load and preprocess the dataset (similar to previous examples)
url = "/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=column_names, sep=',\s*', engine='python')
data = data.dropna()
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

X = data.drop('income', axis=1)
y = data['income']  # Using 'age' as an example for regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if method=='dcst':
    # Create and train a Decision Tree Regression model
    reg = DecisionTreeRegressor(max_depth=5)  # You can adjust max_depth
elif method=='rf':
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators and other hyperparameters
elif method=='svr':
    from sklearn.svm import SVR
    reg = SVR(kernel='linear')
else:
    print('not support!!!')
    os._exit(0)

reg.fit(X_train, y_train)
# Make predictions
y_pred = reg.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree Regression - Mean Squared Error: {mse:.2f}")
print(f"Decision Tree Regression - R-squared: {r2:.2f}")
