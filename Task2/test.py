from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

#Prediction on a sample data object
# Convert categorical columns to string type
categorical_cols = ['offer_type', 'offer_type_of_building', 'city_name', 'voivodeship','market']

#Using inference on given data
new_data = pd.DataFrame({
    'offer_type':['Private'], 
    'offer_type_of_building':['Housing Block'],
    'city_name':['Bolesławiec'],
    'voivodeship':['Lower Silesia'],
    'market':['primary'],
    'floor':[1],
    'area':[27], 
    'rooms':[1]
})

scaler = joblib.load('houses_stdscaler.joblib')
new_data_scaled = scaler.fit_transform(new_data[['rooms', 'floor','area']]) # Apply to numerical columns
new_data_scaled = pd.DataFrame(new_data_scaled, columns=['floor', 'area', 'rooms'])

df1=pd.get_dummies(new_data, columns=categorical_cols)
# print(df1)
# Load the saved OneHotEncoder
transformer=joblib.load("houses_encoder.joblib")

# Apply the encoder to the new data
new_data_encoded = pd.DataFrame(encoder.transform(new_data[categorical_cols]))
new_data_encoded = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))


# Replace values in new_data_encoded with values from df1
new_data_encoded.update(df1)


# Concatenate the encoded categorical variables with the scaled numerical columns
final_data = pd.concat([new_data_scaled, new_data_encoded], axis=1)


# Replace NaN values with 0
final_data = final_data.fillna(0)

model=joblib.load('houses_gb_model.joblib')

#The predicition
predictions=model.predict(final_data)

print(predictions)