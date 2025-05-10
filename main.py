import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import data_frame

# Function to split the dataset into training and test sets
def split_data(data_encoded):
    # Separate features (X) and target (y)
    X = data_encoded[['year', 'week_of_year', 'calls_offered_last_year', 'calls_offered_avg_last_year']
                     + [col for col in data_encoded.columns if col not in ['calls_offered', 'date']]]
    y = data_encoded['calls_offered']

    # Split the data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to train the linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train.columns  # Return model and the column names used during training

# Function to make predictions and calculate metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")
    return y_pred

# Function to save predictions to a CSV file
def save_predictions(X_test, y_test, y_pred, file_name='Forecast.csv'):
    save = input("Would you like to save the predictions to a CSV file? (yes/no): ").strip().lower() 
    if save == 'yes':
        data_with_predictions = X_test.copy()
        data_with_predictions['calls_offered_actual'] = y_test.values
        data_with_predictions['calls_offered_predicted'] = y_pred
        # Save to a CSV file
        data_with_predictions.to_csv(file_name, index=False)
        print(f"CSV file '{file_name}' has been saved to the current directory.")
    else:
        print("Saving skipped. The predictions have not been saved.")

# Function to plot the actual vs predicted calls offered
def plot_predictions(X_test, y_test, y_pred, original_data):
    dates_test = original_data.loc[X_test.index, 'date']
    dates_test = pd.to_datetime(dates_test)
    sorted_indices = dates_test.argsort()
    dates_test = dates_test.iloc[sorted_indices]
    y_test_sorted = y_test.iloc[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test, y_test_sorted, label="Actual Calls Offered", color="blue")
    plt.plot(dates_test, y_pred_sorted, label="Predicted Calls Offered"
             , color="red", linestyle="--")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Calls Offered')
    plt.title('Actual vs Predicted Calls Offered Over Dates (Chronological Order)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Erlang C calculation
def erlang_c(calls_offered, aht=500, target_sl=0.85, max_wait_time=30, shrinkage=0.35):
    traffic_intensity = (calls_offered * aht) / 3600
    staff = math.ceil(traffic_intensity)

    while True:
        occupancy = traffic_intensity / staff
        if occupancy >= 1:
            staff += 1
            continue

        erlang_c = (math.pow(traffic_intensity, staff)
                    / math.factorial(staff)) * (1 / (1 - occupancy))
        erlang_c /= sum(math.pow(traffic_intensity, n)
                        / math.factorial(n) for n in range(staff)) + erlang_c
        pw = erlang_c * math.exp(-(staff - traffic_intensity) * max_wait_time / aht)
        actual_sl = 1 - pw

        if actual_sl >= target_sl:
            break

        staff += 1

    return math.ceil(staff / (1 - shrinkage))


# Function to generate staffing recommendations
def generate_staffing_recommendations(predictions, aht, target_sl, max_wait_time, shrinkage=0.3):
    staffing = []
    for _, row in predictions.iterrows():
        calls_offered = row['calls_offered_predicted']
        staff_needed = erlang_c(calls_offered, aht, target_sl, max_wait_time, shrinkage)
        staffing.append(staff_needed)
    predictions['staff_required'] = staffing
    return predictions


# Function to generate future data for predictions
def generate_future_dates(start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'week_of_year': future_dates.isocalendar().week,
        'week_day': future_dates.strftime('%A')
    })
    return future_data

# Function to align columns with the training data and ensure correct column order
def align_columns_with_training(future_data, training_columns, original_data):
    # One-hot encode the future data as done during training
    future_data_encoded = pd.get_dummies(future_data, columns=['week_day'], drop_first=True)
    # Add missing columns in future data and fill with 0
    for col in training_columns:
        if col not in future_data_encoded.columns:
            future_data_encoded[col] = 0
    # Ensure the order of columns in future data matches the order in training data
    future_data_encoded = future_data_encoded[training_columns]
    return future_data_encoded

# Function to predict future calls_offered based on user-specified date intervals
def predict_future_calls(model, future_data_encoded, future_data, save_future_predictions=False):
    # Predict calls offered for future dates
    future_predictions = model.predict(future_data_encoded)
    # Add predictions to future_data
    future_data['calls_offered_predicted'] = future_predictions
    # Save to CSV if user wants
    if save_future_predictions:
        future_data.to_csv('Future_Forecast.csv', index=False)
        print("Future predictions saved to 'Future_Forecast.csv'")
    return future_data

# Main function
def main():
    file_path = 'phone_data.csv'
    data_encoded, original_data = data_frame.load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data_encoded)
    model, training_columns = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    save_predictions(X_test, y_test, y_pred)
    plot_predictions(X_test, y_test, y_pred, original_data)

    start_date = input("Enter start date for future predictions (YYYY-MM-DD): ")
    end_date = input("Enter end date for future predictions (YYYY-MM-DD): ")
    future_data = pd.date_range(start=start_date, end=end_date)
    future_predictions = pd.DataFrame(
        {'date': future_data
         , 'calls_offered_predicted': y_pred[:len(future_data)]})

    # Average Handling Time (seconds)
    aht = 300
    # Target Service Level (80%)
    target_sl = 0.8
    # Max wait time (seconds)
    max_wait_time = 20
    # Shrinkage factor
    shrinkage = 0.3
    staffed_predictions = generate_staffing_recommendations(
        future_predictions
        , aht
        , target_sl
        , max_wait_time
        , shrinkage)
    print(staffed_predictions)

if __name__ == "__main__":
    main()
