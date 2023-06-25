import pandas as pd
import numpy as np
import re
from datetime import datetime

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
    
# excel_file = 'Dataset_for_test.xlsx'
excel_file = 'https://github.com/amitfallach/Advanced-data-mining-in-Python---FinalProject/blob/main/Dataset_for_test.xlsx'
#data = pd.read_excel(excel_file)
data = pd.read_excel(excel_file, engine='openpyxl')

# excel_file = 'output_all_students_Train_v10.csv'
# data = pd.read_csv(excel_file)

df = prepare_data(df)