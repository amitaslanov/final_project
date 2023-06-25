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


# --------------------------------------------------------------------------------------------------------


# Building a prepare_data function

def prepare_data(df):
    
    # "City" column
    df.loc[:, 'City'] = df['City'].str.replace(' נהריה', 'נהריה')
    df.loc[:, 'City'] = df['City'].str.replace('נהריה', 'נהרייה')
    df.loc[:, 'City'] = df['City'].str.replace(' שוהם','שוהם')
    df.loc[:, 'City'] = df['City'].str.replace('  שוהם','שוהם')
    
    # Clean the "price" column
    df.loc[:, 'price'] = df['price'].astype(str).str.replace('[^\d.]', '', regex=True)
    df.loc[:, 'price'] = pd.to_numeric(df['price'])

    # Drop rows with missing prices
    df.dropna(subset=['price'], inplace=True)
    df.loc[:, 'room_number'] = df['room_number'].astype(str).str.replace('[^\d.]', '', regex=True)
    df.loc[:, 'room_number'] = pd.to_numeric(df['room_number'])

    rows_to_drop = df[df['room_number'] > 10].index

    df = df.drop(rows_to_drop, axis=0)
    
    # "Area" column
    df.loc[:, 'Area'] = df['Area'].astype(str).str.replace('[^\d.]', '', regex=True)
    df.loc[:, 'Area'] = pd.to_numeric(df['Area'])

    # "Street" column
    df.loc[:, 'Street'] = df['Street'].str.replace(r"[\[\]']", '', regex=True)

    # "number_in_street" column
    df.loc[:, 'number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce')

    # "type" column
    df.loc[:, 'type'] = df['type'].str.replace("'", "")

    # "city_area" column
    df.loc[:, 'city_area'] = df['city_area'].astype(str)
    df.loc[:, 'city_area'] = df['city_area'].apply(lambda x: '' if not re.match(r'^[\u0590-\u05FF\s-]+$', x) else x)
    df.loc[:, 'city_area'] = df['city_area'].str.replace('-', '')

    # "floor_out_of" column
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קומה', '')
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קומת ', '')
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('מתוך', ' ')
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('מרתף', '-1')
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קרקע', '0')
    df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('\s+', ' ')

    # "total_floors" and "floor_out_of" columns
    df[['floor', 'total_floors']] = df['floor_out_of'].str.extract(r'(\d+)\D*(\d*)')
    df['floor'] = df.apply(lambda row: row['total_floors'] if pd.isnull(row['floor']) else row['floor'], axis=1)
    df['total_floors'] = df.apply(lambda row: row['floor'] if pd.isnull(row['total_floors']) else row['total_floors'], axis=1)
    df = df.drop('floor_out_of', axis=1)

    # Define the list of types that should result in None for "floor" and "total_floors"
    types_to_ignore = ["בית פרטי", "דו משפחתי", "מגרש", "נחלה", "קוטג'", "קוטג' טורי"]

    # Update "floor" and "total_floors" to None for the specified types
    df.loc[df['type'].isin(types_to_ignore), ['floor', 'total_floors']] = 'None'

    # "has ___" columns
    columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly ']
    for column in columns_to_convert:
        df[column] = df[column].apply(lambda x: 0 if pd.isna(x) else (1 if x in [True, 'true'] else (0 if x in [False, 'false', 'אין', 'לא', 'no'] else (1 if any(word in str(x) for word in ['יש', 'כן', 'yes']) else 0))))

    # "entranceDate" column
    df['entranceDate '] = df['entranceDate '].replace({
        'גמיש': 'flexible',
        'גמיש ': 'flexible',
        'לא צויין': 'not_defined',
        'מיידי': 'immediate'
    })

    df['days'] = ""

    date_format = "%Y-%m-%d"  

    for index, row in df.iterrows():
        entrance_date = row['entranceDate ']
    
        if isinstance(entrance_date, datetime):
            formatted_date = entrance_date.strftime(date_format)
        
            days = (entrance_date - datetime.now()).days
            df.at[index, 'entranceDate '] = formatted_date
            df.at[index, 'days'] = days
        
            if days < 180 or df.at[index, 'entranceDate '] == 'immediate':
                df.at[index, 'entranceDate '] = "less_than_6 months"
            elif 180 <= days <= 365:
                df.at[index, 'entranceDate '] = "months_6_12"
            else:
                df.at[index, 'entranceDate '] = "above_year"
            
    df = df.drop('days', axis=1)

    # "condition" column
    df['condition '] = df['condition '].str.replace('FALSE', 'None')
    df['condition '] = df['condition '].str.replace('לא צויין', 'None')
    df['condition '].fillna('None', inplace=True)

    # "publishedDays" column
    df['publishedDays '] = df['publishedDays '].apply(lambda x: re.sub('[^\d]+', '', str(x)) if pd.notnull(x) else '')
    df['publishedDays '] = pd.to_numeric(df['publishedDays '], errors='coerce')

    #droping rows that the price is less then 500k (probably an input error)
    df.drop(df[df['price'] < 500000].index, inplace=True)

    #fill in missing values
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce')
    df['floor'].fillna(0, inplace=True)
    df['total_floors'].fillna(0, inplace=True)

    df['num_of_images'].fillna(0, inplace=True)

    mean_value_room_number = df['room_number'].mean()
    df['room_number'].fillna(mean_value_room_number, inplace=True)

    mean_value_area = df['Area'].mean()
    df['Area'].fillna(mean_value_area, inplace=True)

    mean_value_number_in_street = df['number_in_street'].mean()
    df['number_in_street'].fillna(mean_value_number_in_street, inplace=True)

    mean_value_publishedDays = df['publishedDays '].mean()
    df['publishedDays '].fillna(mean_value_publishedDays, inplace=True)
    
    df = df.drop(['Street', 'city_area', 'description '], axis=1)

    # delete unnecessary spaces
    df.columns = df.columns.str.replace(' ', '')

    # Check for duplicate rows
    duplicate_rows = df.duplicated()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)
    
    return df

# --------------------------------------------------------------------------------------------------------

# Importing The Data 

url = "https://github.com/amitaslanov/final_project/raw/main/output_all_students_Train_v10.xlsx"

xls = pd.ExcelFile(url)
df = xls.parse(0)

# Use the 'prepare_data' function to process the data

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