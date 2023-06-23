{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "e52d2ac7-5677-4169-a691-71b83222e2cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "from datetime import datetime\n",
    "\n",
    "url = \"https://github.com/amitaslanov/final_project/raw/main/output_all_students_Train_v10.xlsx\"\n",
    "\n",
    "# Read the Excel file from the URL\n",
    "xls = pd.ExcelFile(url)\n",
    "\n",
    "# Parse the first sheet of the Excel file\n",
    "df = xls.parse(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0eea03d-de80-4685-a98f-1d7f336e4812",
   "metadata": {},
   "source": [
    "#### Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "50c11f8f-c67d-4329-87f6-60c117e58f41",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\amits\\AppData\\Local\\Temp\\ipykernel_16088\\2939205284.py:45: FutureWarning: The default value of regex will change from True to False in a future version.\n",
      "  df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('\\s+', ' ')\n"
     ]
    }
   ],
   "source": [
    "# \"City\" column\n",
    "df.loc[:, 'City'] = df['City'].str.replace(' נהריה', 'נהריה')\n",
    "df.loc[:, 'City'] = df['City'].str.replace('נהריה', 'נהרייה')\n",
    "df.loc[:, 'City'] = df['City'].str.replace(' שוהם','שוהם')\n",
    "df.loc[:, 'City'] = df['City'].str.replace('  שוהם','שוהם')\n",
    "\n",
    "# Clean the \"price\" column\n",
    "df.loc[:, 'price'] = df['price'].astype(str).str.replace('[^\\d.]', '', regex=True)\n",
    "df.loc[:, 'price'] = pd.to_numeric(df['price'])\n",
    "\n",
    "# Drop rows with missing prices\n",
    "df.dropna(subset=['price'], inplace=True)\n",
    "# Clean the \"room_number\" column\n",
    "df.loc[:, 'room_number'] = df['room_number'].astype(str).str.replace('[^\\d.]', '', regex=True)\n",
    "df.loc[:, 'room_number'] = pd.to_numeric(df['room_number'])\n",
    "\n",
    "rows_to_drop = df[df['room_number'] > 10].index\n",
    "\n",
    "# Drop the identified rows\n",
    "df = df.drop(rows_to_drop, axis=0)\n",
    "# \"Area\" column\n",
    "df.loc[:, 'Area'] = df['Area'].astype(str).str.replace('[^\\d.]', '', regex=True)\n",
    "df.loc[:, 'Area'] = pd.to_numeric(df['Area'])\n",
    "\n",
    "# \"Street\" column\n",
    "df.loc[:, 'Street'] = df['Street'].str.replace(r\"[\\[\\]']\", '', regex=True)\n",
    "\n",
    "# \"number_in_street\" column\n",
    "df.loc[:, 'number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce')\n",
    "\n",
    "# \"type\" column\n",
    "df.loc[:, 'type'] = df['type'].str.replace(\"'\", \"\")\n",
    "\n",
    "# \"city_area\" column \n",
    "df.loc[:, 'city_area'] = df['city_area'].astype(str)\n",
    "df.loc[:, 'city_area'] = df['city_area'].apply(lambda x: '' if not re.match(r'^[\\u0590-\\u05FF\\s-]+$', x) else x)\n",
    "df.loc[:, 'city_area'] = df['city_area'].str.replace('-', '')\n",
    "\n",
    "# \"floor_out_of\" column\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קומה', '')\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קומת ', '')\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('מתוך', ' ')\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('מרתף', '-1')\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('קרקע', '0')\n",
    "df.loc[:, 'floor_out_of'] = df['floor_out_of'].str.replace('\\s+', ' ')\n",
    "\n",
    "# \"total_floors\" and \"floor_out_of\" columns\n",
    "df[['floor', 'total_floors']] = df['floor_out_of'].str.extract(r'(\\d+)\\D*(\\d*)')\n",
    "df['floor'] = df.apply(lambda row: row['total_floors'] if pd.isnull(row['floor']) else row['floor'], axis=1)\n",
    "df['total_floors'] = df.apply(lambda row: row['floor'] if pd.isnull(row['total_floors']) else row['total_floors'], axis=1)\n",
    "df = df.drop('floor_out_of', axis=1)\n",
    "\n",
    "# Define the list of types that should result in None for \"floor\" and \"total_floors\"\n",
    "types_to_ignore = [\"בית פרטי\", \"דו משפחתי\", \"מגרש\", \"נחלה\", \"קוטג'\", \"קוטג' טורי\"]\n",
    "\n",
    "# Update \"floor\" and \"total_floors\" to None for the specified types\n",
    "df.loc[df['type'].isin(types_to_ignore), ['floor', 'total_floors']] = 'None'\n",
    "# \"has ___\" columns\n",
    "columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly ']\n",
    "for column in columns_to_convert:\n",
    "    df[column] = df[column].apply(lambda x: 0 if pd.isna(x) else (1 if x in [True, 'true'] else (0 if x in [False, 'false', 'אין', 'לא', 'no'] else (1 if any(word in str(x) for word in ['יש', 'כן', 'yes']) else 0))))\n",
    "\n",
    "# \"entranceDate\" column\n",
    "df['entranceDate '] = df['entranceDate '].replace({\n",
    "    'גמיש': 'flexible',\n",
    "    'גמיש ': 'flexible',\n",
    "    'לא צויין': 'not_defined',\n",
    "    'מיידי': 'immediate'\n",
    "})\n",
    "\n",
    "\n",
    "df['days'] = \"\"\n",
    "\n",
    "date_format = \"%Y-%m-%d\"  \n",
    "\n",
    "for index, row in df.iterrows():\n",
    "    entrance_date = row['entranceDate ']\n",
    "    \n",
    "    if isinstance(entrance_date, datetime):\n",
    "        formatted_date = entrance_date.strftime(date_format)\n",
    "        \n",
    "        days = (entrance_date - datetime.now()).days\n",
    "        df.at[index, 'entranceDate '] = formatted_date\n",
    "        df.at[index, 'days'] = days\n",
    "        \n",
    "        if days < 180 or df.at[index, 'entranceDate '] == 'immediate':\n",
    "            df.at[index, 'entranceDate '] = \"less_than_6 months\"\n",
    "        elif 180 <= days <= 365:\n",
    "            df.at[index, 'entranceDate '] = \"months_6_12\"\n",
    "        else:\n",
    "            df.at[index, 'entranceDate '] = \"above_year\"\n",
    "            \n",
    "df = df.drop('days', axis=1)\n",
    "\n",
    "# \"condition\" column\n",
    "df['condition '] = df['condition '].str.replace('FALSE', 'None')\n",
    "df['condition '] = df['condition '].str.replace('לא צויין', 'None')\n",
    "df['condition '].fillna('None', inplace=True)\n",
    "\n",
    "# \"publishedDays\" column\n",
    "df['publishedDays '] = df['publishedDays '].apply(lambda x: re.sub('[^\\d]+', '', str(x)) if pd.notnull(x) else '')\n",
    "df['publishedDays '] = pd.to_numeric(df['publishedDays '], errors='coerce')\n",
    "\n",
    "#droping rows that the price is less then 500k (probably an input error)\n",
    "df.drop(df[df['price'] < 500000].index, inplace=True)\n",
    "\n",
    "#fill in missing values\n",
    "df['floor'] = pd.to_numeric(df['floor'], errors='coerce')\n",
    "df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce')\n",
    "df['floor'].fillna(0, inplace=True)\n",
    "df['total_floors'].fillna(0, inplace=True)\n",
    "\n",
    "df['num_of_images'].fillna(0, inplace=True)\n",
    "\n",
    "mean_value_room_number = df['room_number'].mean()\n",
    "df['room_number'].fillna(mean_value_room_number, inplace=True)\n",
    "\n",
    "mean_value_area = df['Area'].mean()\n",
    "df['Area'].fillna(mean_value_area, inplace=True)\n",
    "\n",
    "mean_value_number_in_street = df['number_in_street'].mean()\n",
    "df['number_in_street'].fillna(mean_value_number_in_street, inplace=True)\n",
    "\n",
    "mean_value_publishedDays = df['publishedDays '].mean()\n",
    "df['publishedDays '].fillna(mean_value_publishedDays, inplace=True)\n",
    "df = df.drop(['Street', 'city_area', 'description '], axis=1)\n",
    "\n",
    "# Check for duplicate rows\n",
    "duplicate_rows = df.duplicated()\n",
    "\n",
    "# Remove duplicate rows\n",
    "df = df.drop_duplicates()\n",
    "\n",
    "# Optional: Reset the index of the DataFrame\n",
    "df = df.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "1baf6671-9eeb-49fa-a269-92c2d6e814a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['פתח תקווה', 'נתניה', 'באר שבע', 'הרצליה', 'אריאל', 'דימונה',\n",
       "       'רחובות', 'גבעת שמואל', 'ירושלים', 'שוהם', 'כפר סבא', 'רעננה',\n",
       "       'נהרייה', 'זכרון יעקב', 'קרית ביאליק', 'חיפה', 'הוד השרון',\n",
       "       'תל אביב', 'ראשון לציון', 'יהוד מונוסון', 'נס ציונה', 'אילת',\n",
       "       'חולון', 'מודיעין מכבים רעות', 'צפת', 'בת ים', 'רמת גן',\n",
       "       'נוף הגליל', 'בית שאן'], dtype=object)"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['City'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ebbb174-4504-4e48-850d-9aa31157b583",
   "metadata": {},
   "source": [
    "#### One Hot Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "6b1920b2-19fe-44d6-9665-4b05c6109dcf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Columns to perform one-hot encoding on\n",
    "columns_to_encode = ['City', 'type', 'condition ', 'entranceDate ', 'furniture ']\n",
    "\n",
    "# Perform one-hot encoding\n",
    "encoded_df = pd.get_dummies(df, columns=columns_to_encode)\n",
    "encoded_df.columns = encoded_df.columns.str.replace(' ', '')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76e0c790-2e36-4570-a40d-8ccebb093957",
   "metadata": {},
   "source": [
    "#### Features Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "959904b7-77f6-49fd-b901-5e7caeb70bd4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected Features:\n",
      "room_number\n",
      "hasElevator\n",
      "hasStorage\n",
      "hasAirCondition\n",
      "hasBalcony\n",
      "handicapFriendly\n",
      "City_אילת\n",
      "City_ביתשאן\n",
      "City_בתים\n",
      "City_גבעתשמואל\n",
      "City_הודהשרון\n",
      "City_הרצליה\n",
      "City_חולון\n",
      "City_ירושלים\n",
      "City_כפרסבא\n",
      "City_צפת\n",
      "City_קריתביאליק\n",
      "City_רחובות\n",
      "City_שוהם\n",
      "City_תלאביב\n",
      "type_אחר\n",
      "type_בניין\n",
      "type_דומשפחתי\n",
      "type_דופלקס\n",
      "type_דירה\n",
      "type_דירתגן\n",
      "type_טריפלקס\n",
      "type_מגרש\n",
      "type_מיניפנטהאוז\n",
      "type_נחלה\n",
      "type_קוטגטורי\n",
      "condition_None\n",
      "condition_דורששיפוץ\n",
      "condition_חדש\n",
      "condition_ישן\n",
      "condition_משופץ\n",
      "condition_שמור\n",
      "furniture_אין\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Assuming features_selection_data is your feature matrix and 'price' is the target variable\n",
    "X = encoded_df.drop('price', axis=1)\n",
    "y = encoded_df['price']\n",
    "\n",
    "# Create the linear regression model (or choose a different model)\n",
    "model = LinearRegression()\n",
    "\n",
    "# Create the RFE object with automatic feature selection\n",
    "rfe = RFE(estimator=model, n_features_to_select=None)\n",
    "\n",
    "# Fit the RFE model to the data\n",
    "rfe.fit(X, y)\n",
    "\n",
    "# Get the selected feature indices\n",
    "selected_indices = rfe.support_\n",
    "\n",
    "# Get the selected feature names\n",
    "selected_features = X.columns[selected_indices]\n",
    "\n",
    "# Print the selected feature names\n",
    "print(\"Selected Features:\")\n",
    "for feature in selected_features:\n",
    "    print(feature)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "74076f84-58d1-4b46-9f61-c05ec327052f",
   "metadata": {},
   "outputs": [],
   "source": [
    "selected_features = selected_features.tolist()\n",
    "selected_features.append('room_number')\n",
    "selected_features.append('Area')\n",
    "selected_features.remove('furniture_אין')\n",
    "selected_features = pd.Index(selected_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "a845bb3c-e9d7-47a2-a16a-6af3bed36687",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['room_number', 'hasElevator', 'hasStorage', 'hasAirCondition',\n",
       "       'hasBalcony', 'handicapFriendly', 'City_אילת', 'City_ביתשאן',\n",
       "       'City_בתים', 'City_גבעתשמואל', 'City_הודהשרון', 'City_הרצליה',\n",
       "       'City_חולון', 'City_ירושלים', 'City_כפרסבא', 'City_צפת',\n",
       "       'City_קריתביאליק', 'City_רחובות', 'City_שוהם', 'City_תלאביב',\n",
       "       'type_אחר', 'type_בניין', 'type_דומשפחתי', 'type_דופלקס', 'type_דירה',\n",
       "       'type_דירתגן', 'type_טריפלקס', 'type_מגרש', 'type_מיניפנטהאוז',\n",
       "       'type_נחלה', 'type_קוטגטורי', 'condition_None', 'condition_דורששיפוץ',\n",
       "       'condition_חדש', 'condition_ישן', 'condition_משופץ', 'condition_שמור',\n",
       "       'room_number', 'Area'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "selected_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "b9fdc9b5-e73c-437f-a970-6fc3ad063d3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 3460591997056.127\n",
      "Mean Absolute Error: 1499137.4468417477\n",
      "R^2 Score: -0.08033070829953592\n",
      "Cross-Validation RMSE Scores: [ 1468197.13834565 48169095.51831906  1935255.83353739  1714955.79007963\n",
      "  2136879.09811112  1760686.28973379  2081090.34565001  2314148.86294451\n",
      "  1548544.00735459  2193657.82158172]\n",
      "Mean Cross-Validation RMSE: 6532251.070565747\n",
      "Standard Deviation of Cross-Validation RMSE: 13881510.103596123\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import ElasticNet\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "import numpy as np\n",
    "\n",
    "# Assuming selected_features contains the list of selected feature names\n",
    "X = encoded_df[selected_features]\n",
    "y = encoded_df['price']\n",
    "\n",
    "# Split the data into train and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Create and fit the Elastic Net model\n",
    "model = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Predict on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Calculate evaluation metrics\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(\"Mean Squared Error:\", mse)\n",
    "print(\"Mean Absolute Error:\", mae)\n",
    "print(\"R^2 Score:\", r2)\n",
    "\n",
    "# Perform cross-validation\n",
    "cv_scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')\n",
    "rmse_scores = np.sqrt(-cv_scores)\n",
    "mean_rmse = np.mean(rmse_scores)\n",
    "std_rmse = np.std(rmse_scores)\n",
    "\n",
    "print(\"Cross-Validation RMSE Scores:\", rmse_scores)\n",
    "print(\"Mean Cross-Validation RMSE:\", mean_rmse)\n",
    "print(\"Standard Deviation of Cross-Validation RMSE:\", std_rmse)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05b01951-e5fa-45e5-86b5-2a0039935b80",
   "metadata": {},
   "source": [
    "#### Elastic Net Prediction Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "93455609-7ca8-48e8-9e81-45379332b8e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['trained_model.pkl']"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import ElasticNet\n",
    "import joblib\n",
    "\n",
    "X = encoded_df[selected_features]\n",
    "y = encoded_df['price']\n",
    "\n",
    "# Create and fit the Elastic Net model\n",
    "model = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed\n",
    "model.fit(X, y)\n",
    "\n",
    "# Save the trained model as a PKL file\n",
    "joblib.dump(model, 'trained_model.pkl')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
