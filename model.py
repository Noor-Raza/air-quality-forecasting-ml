import boto3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### Access AWS

# AWS credentials and bucket details
aws_access_key = "your_access_key"
aws_secret_key = "your_secret_key"

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)



### Load Data

# File keys and download path
bucket_name = 'bcnairquality'
keys = ['202406_airquality_data.csv',
        '202407_airquality_data.csv',
        '202404_airquality_data.csv',
        '202403_airquality_data.csv',
        '202402_airquality_data.csv',
        '202401_airquality_data.csv',
        '202408_airquality_data.csv',
        '202411_airquality_data.csv',
        '202409_airquality_data.csv',
        '202410_airquality_data.csv',
        '202412_airquality_data.csv',
        '2023_weather_data.csv',
        '2024_weather_data.csv']
download_path = '/tmp/'

# Download files
for key in keys:
    try:
        s3.download_file(bucket_name, key, f"{download_path}{key}")
        print(f"Successfully downloaded {key}")
    except Exception as e:
        print(f"Error downloading {key}: {e}")

# Load files
raw_data = {}

try:
    for key in keys:
        raw_data[key] = pd.read_csv(f"{download_path}{key}")
    print(f"Successfully loaded datasets")
except Exception as e:
    print(f"Error processing {keys[0]}: {e}")



### Combine airquality datasets

# Concatenate datasets
airquality_df = pd.concat([raw_data[key] for key in keys[:-2]])

# Create a 'date' column using the 'ANY', 'MES', and 'DIA' columns
airquality_df['DATE'] = pd.to_datetime(
    airquality_df[['ANY', 'MES', 'DIA']].astype(str).agg('-'.join, axis=1), errors='coerce'
)

## Print update
print("Successfully combined airquality data")



### Combine weather datasets

# Strip quotes from column names
for key in keys[-2:]:
    raw_data[key].columns = raw_data[key].columns.str.strip('"')  # Remove quotes around column names

# Concatenate datasets
weather_df = pd.concat([raw_data[key] for key in keys[-2:]])

# Create a 'date' column
weather_df['DATE'] = pd.to_datetime(weather_df['DATA_LECTURA'], errors='coerce')

# Strip quotes from rows
weather_df = weather_df.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)

## Print update
print("Successfully combined weather data")



### Merge datasets

# Aggregate weather data by day and acronym
weather_daily = weather_df.groupby(['DATE', 'ACRÒNIM']).agg({'VALOR': 'mean'}).reset_index()

# Pivot weather data to have one column per acronym
weather_daily_pivot = weather_daily.pivot(index='DATE', columns='ACRÒNIM', values='VALOR').reset_index()

# Reshape air quality data for hourly analysis
new_airquality_df = airquality_df.melt(
    id_vars=['DATE', 'ESTACIO', 'CODI_CONTAMINANT'],
    value_vars=[f'H{str(i).zfill(2)}' for i in range(1, 25)],
    var_name='HOUR',
    value_name='POLLUTION'
)

# Aggregate air quality data by day and pollutant type
airquality_no2 = new_airquality_df[new_airquality_df['CODI_CONTAMINANT'] == 8].groupby('DATE')['POLLUTION'].max().reset_index().rename(columns={'POLLUTION': 'MAX_NO2'})
airquality_pm10 = new_airquality_df[new_airquality_df['CODI_CONTAMINANT'] == 110].groupby('DATE')['POLLUTION'].mean().reset_index().rename(columns={'POLLUTION': 'AVG_PM10'})

# Combine datasets for both pollutants
airquality_daily_pivot = pd.merge(airquality_no2, airquality_pm10, on='DATE', how='outer')

# Merge airquality and weather datasets on date
daily_merged_df = pd.merge(weather_daily_pivot, airquality_daily_pivot, on='DATE', how='inner')

# Sort merged data by date column
data = daily_merged_df.sort_values(by='DATE')
data.reset_index(drop=True, inplace=True)

## Print update
print("Successfully merged datasets")



### Preprocess data

# Remove N/As
data = data.dropna()

# Derive temporal features
data['Day_of_Week'] = data['DATE'].dt.dayofweek  # Monday = 0, Sunday = 6
data['Is_Weekend'] = data['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend = 1
data['Month'] = data['DATE'].dt.month  # Month as a number
data['Is_Spring'] = data['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)  # Spring = 1
data['Is_Summer'] = data['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)  # Summer = 1
data['Is_Autumn'] = data['Month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)  # Autumn = 1
data['Is_Winter'] = data['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)  # Winter = 1
data['Day_of_Year'] = data['DATE'].dt.dayofyear  # Day of the year (1-365)
data['Week_of_Year'] = data['DATE'].dt.isocalendar().week  # Week number of the year

# Define feature columns
weather_columns = weather_daily_pivot.columns.to_list()[1:]
temporal_columns = ['Day_of_Week', 'Is_Weekend', 'Month', 'Is_Spring', 'Is_Summer', 'Is_Autumn', 'Is_Winter', 'Day_of_Year', 'Week_of_Year']

## Print update
print("Successfully preprocessed data")



### Define Model

# Define pipeline functions
scaler = StandardScaler()
model = RandomForestRegressor(random_state=42)

# Machine learning pipeline
pipe = Pipeline(steps=[
    ('scaler', scaler),
    ('regressor', model)
])

# Grid search parameters
param_grid = [
    {
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5, 10]
    },
    {
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 10]

    },
    {
        'regressor': [SVR()],
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['linear', 'rbf', 'poly'],
        'regressor__epsilon': [0.1, 0.2, 0.5]
    },
    {
        'regressor': [ElasticNet(random_state=42)],
        'regressor__alpha': [0.1, 1, 10],
        'regressor__l1_ratio': [0.1, 0.5, 0.9]
    }
]

# Grid search with cross-validation
grid = GridSearchCV(pipe, param_grid, cv=5, verbose=1, scoring='r2')

## Print update
print("Successfully trained model")



### Fit NO2 Model

# Features and target selection
X_NO2 = data[weather_columns + temporal_columns]
y_NO2 = data['MAX_NO2']

# Split data into training and test sets (80% training, 20% test)
X_train_NO2, X_test_NO2, y_train_NO2, y_test_NO2 = train_test_split(X_NO2, y_NO2, test_size=0.2, random_state=42)

# Fit the model
grid.fit(X_train_NO2, y_train_NO2)

# Select best model and generalization score
best_model_NO2 = grid.best_estimator_

# Predict y
y_pred_NO2 = best_model_NO2.predict(X_test_NO2)

# Evaluate model
r2_NO2 = r2_score(y_test_NO2, y_pred_NO2)
mae_NO2 = mean_absolute_error(y_test_NO2, y_pred_NO2)
mse_NO2 = mean_squared_error(y_test_NO2, y_pred_NO2)

## Print update
print(f"Successfully fitted NO2 (R2: {r2_NO2:.2f}; MAE: {mae_NO2:.2f}; MSE: {mse_NO2:.2f})")



### Fit PM10 Model

# Features and target selection
X_PM10 = data[weather_columns + temporal_columns]
y_PM10 = data['AVG_PM10']

# Split data into training and test sets (80% training, 20% test)
X_train_PM10, X_test_PM10, y_train_PM10, y_test_PM10 = train_test_split(X_PM10, y_PM10, test_size=0.2, random_state=42)

# Fit the model
grid.fit(X_train_PM10, y_train_PM10)

# Select best model and generalization score
best_model_PM10 = grid.best_estimator_

# Predict y
y_pred_PM10 = best_model_PM10.predict(X_test_PM10)

# Evaluate model
r2_PM10 = r2_score(y_test_PM10, y_pred_PM10)
mae_PM10 = mean_absolute_error(y_test_PM10, y_pred_PM10)
mse_PM10 = mean_squared_error(y_test_PM10, y_pred_PM10)

## Print update
print(f"Successfully fitted PM10 (R2: {r2_PM10:.2f}; MAE: {mae_PM10:.2f}; MSE: {mse_PM10:.2f})")



### Get predictions

# Function to classify the thresholds for NO2 and PM10 (Source: https://ajuntament.barcelona.cat/qualitataire/en/air-quality/how-we-are-fighting-against-pollution/atmospheric-pollution-monitoring-and-forecasting)
def classify_no2(no2):
    if no2 <= 40:
        return 0
    elif no2 <= 90:
        return 1
    elif no2 <= 120:
        return 2
    elif no2 <= 230:
        return 3
    elif no2 <= 340:
        return 4
    else:
        return 5

def classify_pm10(pm10):
    if pm10 <= 20:
        return 0
    elif pm10 <= 40:
        return 1
    elif pm10 <= 50:
        return 2
    elif pm10 <= 100:
        return 3
    elif pm10 <= 150:
        return 4
    else:
        return 5

# Function to calculate the average weather for a given future date
def get_average_weather_for_future_date(day, month, year):
    input_date = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    target_date = pd.to_datetime(input_date)
    prev_year_date = target_date.replace(year=target_date.year - 1)
    date_range = pd.date_range(prev_year_date - pd.Timedelta(days=1),
                               prev_year_date + pd.Timedelta(days=1), freq='D')
    weather_data = weather_daily_pivot.loc[weather_daily_pivot['DATE'].dt.date.isin(date_range.date)]

    if len(weather_data) < len(date_range):
        return None

    avg_weather = weather_data.mean()
    return avg_weather

# Function to get the predicted air pollution for a specific date
def predict_air_quality(day, month, year):
    try:
        avg_weather = get_average_weather_for_future_date(day, month, year)
        input_date = pd.Timestamp(year=int(year), month=int(month), day=int(day))

        if avg_weather is None:
            return "Unable to calculate weather for the given future date."

        features = {}
        weather_features = avg_weather[weather_columns]
        temporal_features = {
            'Day_of_Week': input_date.dayofweek,
            'Is_Weekend': 1 if input_date.weekday() >= 5 else 0,
            'Month': input_date.month,
            'Is_Spring': 1 if input_date.month in [3, 4, 5] else 0,
            'Is_Summer': 1 if input_date.month in [6, 7, 8] else 0,
            'Is_Autumn': 1 if input_date.month in [9, 10, 11] else 0,
            'Is_Winter': 1 if input_date.month in [12, 1, 2] else 0,
            'Day_of_Year': input_date.dayofyear,
            'Week_of_Year': input_date.isocalendar()[1]
        }
        features = {**weather_features, **temporal_features}
        features_df = pd.DataFrame([features], columns=features.keys())

        NO2_pred = best_model_NO2.predict(features_df)
        PM10_pred = best_model_PM10.predict(features_df)

        NO2_class = classify_no2(NO2_pred[0])
        PM10_class = classify_pm10(PM10_pred[0])

        EQAB_rank = max(NO2_class, PM10_class)

        conditions = ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"]
        health_impacts = [
            "Air quality is satisfactory, and air pollution poses little or no risk.",
            "Air quality is acceptable; however, some pollutants may pose a moderate health concern for a very small number of individuals who are unusually sensitive to air pollution.",
            "Members of sensitive groups may experience health effects; the general public is less likely to be affected.",
            "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
            "Alert; The risk of health effects is increased for everyone.",
            "Health warnings of emergency conditions; the entire population is more likely to be affected."
        ]
        recommendations = [
            "No specific actions needed; normal outdoor activities can be enjoyed.",
            "Sensitive individuals should consider limiting prolonged outdoor exertion.",
            "Sensitive groups (e.g., children, elderly, individuals with respiratory or heart conditions) should reduce prolonged or heavy outdoor exertion.",
            "Sensitive groups should avoid prolonged or heavy outdoor exertion; the general public should reduce prolonged or heavy outdoor exertion.",
            "Sensitive groups should avoid all outdoor exertion; the general public should avoid prolonged or heavy outdoor exertion.",
            "Everyone should avoid all outdoor exertion; follow any additional advice from health authorities."
        ]
        
        return_date = input_date.strftime('%d. %B %Y')
        return_condition = conditions[EQAB_rank]
        return_health_impact = health_impacts[EQAB_rank]
        return_recommendation = recommendations[EQAB_rank]
        return_max_NO2 = round(float(NO2_pred[0]), ndigits=2)
        return_avg_PM10 = round(float(PM10_pred[0]), ndigits=2)
        return_avg_temperature = avg_weather['TM']
        return_max_temperature = avg_weather['TX']
        return_min_temperature = avg_weather['TN']
        return_avg_humidity = avg_weather['HRM']
        return_max_humidity = avg_weather['HRX']
        return_min_humidity = avg_weather['HRN']
        return_avg_atmosphere = avg_weather['PM']
        return_max_atmosphere = avg_weather['PX']
        return_min_atmosphere = avg_weather['PN']
        return_percipitation = avg_weather['PPT']
        return_avg_wind = avg_weather['VVM10']
        return_max_wind = avg_weather['VVX10']

        return f"""
        On {return_date} the air quality will be {return_condition}.

        Health impact: {return_health_impact}
        Recommendation: {return_recommendation}
        
        Predicted pollutants:
         - Max. NO2: {return_max_NO2}
         - Avg. PM10: {return_avg_PM10}
        
        Weather forecast:
         - Temperature: Avg. {return_avg_temperature:.2f}°C; Max. {return_max_temperature:.2f}°C; Min. {return_min_temperature:.2f}°C
         - Humidity: Avg. {return_avg_humidity:.2f}%; Max. {return_max_humidity:.2f}%; Min. {return_min_humidity:.2f}%
         - Atmospheric pressure: Avg. {return_avg_atmosphere:.2f}hPa; Max. {return_max_atmosphere:.2f}hPa; Min. {return_min_atmosphere:.2f}hPa
         - Percipitation: Cum. {return_percipitation:.2f}mm
         - Wind: Avg. {return_avg_wind:.2f}m/s; Max. {return_max_wind:.2f}m/s
        """

    except Exception as e:
        return f"Error in prediction: {str(e)}"
