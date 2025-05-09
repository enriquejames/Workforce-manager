import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

#Use this file to import and load data into data frame and split the training and test data

# Load and preprocess the dataset, including historical features
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Fill missing 'calls_offered' values with 0
    data['calls_offered'] = data['calls_offered'].fillna(0)

    # Add historical features (calls offered from the previous year)
    data['date'] = pd.to_datetime(data['date'])
    data = add_historical_features(data)
    
    # One-hot encode categorical variables (month, week_day, phone_queue)
    data_encoded = pd.get_dummies(data, columns=['month', 'week_day', 'phone_queue'], drop_first=True)

    # Convert 'service_level' from percentage strings to float values
    data_encoded['service_level'] = data_encoded['service_level'].str.rstrip('%').astype(float)

    # Fill any remaining missing values with 0
    data_encoded.fillna(0, inplace=True)

    return data_encoded, data

# Function to add historical call volume features
def add_historical_features(data):
    data['year'] = data['date'].dt.year
    data['day_of_year'] = data['date'].dt.dayofyear
    data['day_of_week'] = data['date'].dt.dayofweek
    
    # Create 'calls_offered_last_year' feature
    data['calls_offered_last_year'] = data.apply(lambda row: 
        data[(data['day_of_year'] == row['day_of_year']) & (data['year'] == row['year'] - 1)]['calls_offered'].mean(), axis=1)
    
    # Create 'calls_offered_avg_last_year' feature (average of similar days in previous year)
    data['calls_offered_avg_last_year'] = data.apply(lambda row: 
        data[(data['day_of_week'] == row['day_of_week']) & (data['year'] == row['year'] - 1)]['calls_offered'].mean(), axis=1)
    
    # Fill NaN values in historical features with 0 (if no matching data from last year)
    data['calls_offered_last_year'].fillna(0, inplace=True)
    data['calls_offered_avg_last_year'].fillna(0, inplace=True)
    
    return data
