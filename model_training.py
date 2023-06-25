import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from madlan_data_prep import prepare_data

url = "https://github.com/amitaslanov/final_project/raw/main/output_all_students_Train_v10.xlsx"

xls = pd.ExcelFile(url)
df = xls.parse(0)

df = prepare_data(df)


# --------------------------------------------------------------------------------------------------------


# One Hot Encoding 

columns_to_encode = ['City', 'type', 'condition', 'entranceDate', 'furniture']

encoded_df = pd.get_dummies(df, columns=columns_to_encode)


# --------------------------------------------------------------------------------------------------------


# Features Selection

X = encoded_df.drop('price', axis=1)
y = encoded_df['price']

# Create the linear regression model
model = LinearRegression()

# Create the RFE object with automatic feature selection
rfe = RFE(estimator=model, n_features_to_select=None)

# Fit the RFE model to the data
rfe.fit(X, y)

# Get the selected feature indices
selected_indices = rfe.support_

# Get the selected feature names
selected_features = X.columns[selected_indices]

# Print the selected feature names
print("Selected Features:")
for feature in selected_features:
    print(feature)

# We decided to drop the street and city_area columns
# because there is alot of different streests and areas in every city,
# and if we will do one-hot-encoding to these column we will get a huge data.
# From our knowledge these details has a very minor effect on the price.

# We decided to drop the details column because she is not categorial
# and the important details the written in it provided in the othe columns (rooms/area and more).

selected_features = selected_features.tolist()
selected_features.append('room_number')
selected_features.append('Area')
selected_features.remove('furniture_אין')
selected_features = pd.Index(selected_features)


# --------------------------------------------------------------------------------------------------------


# Elastic Net Prediction Model

X = encoded_df[selected_features]
y = encoded_df['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Elastic Net model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean Cross-Validation RMSE:", mean_rmse)
print("Standard Deviation of Cross-Validation RMSE:", std_rmse)


# --------------------------------------------------------------------------------------------------------


# Creating trained model pkl file

X = encoded_df[selected_features]
y = encoded_df['price']

# Create and fit the Elastic Net model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)
model.fit(X, y)

# Save the trained model as a PKL file
joblib.dump(model, 'trained_model.pkl')