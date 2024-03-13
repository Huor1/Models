import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


#Makes predictions
def predict(scaler,encoder ,model,input_data):
    
     # Convert categorical columns to string type
    categorical_cols = ['offer_type', 'offer_type_of_building', 'city_name', 'voivodeship','market']

    #This is a sample on how the input dataframe should look like
    # new_data = pd.DataFrame({
    #     'offer_type':['Private'], 
    #     'offer_type_of_building':['Housing Block'],
    #     'city_name':['Bolesławiec'],
    #     'voivodeship':['Lower Silesia'],
    #     'market':['primary'],
    #     'floor':[1],
    #     'area':[27], 
    #     'rooms':[1]
    # })

    
    input_data_scaled = scaler.fit_transform(input_data[['rooms', 'floor','area']]) # Apply to numerical columns
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=['floor', 'area', 'rooms'])

    df1=pd.get_dummies(input_data, columns=categorical_cols)
    
    # Apply the encoder to the new data
    input_data_encoded = pd.DataFrame(encoder.transform(input_data[categorical_cols]))
    input_data_encoded = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))


    # Replace values in new_data_encoded with values from df1
    input_data_encoded.update(df1)


    # Concatenate the encoded categorical variables with the scaled numerical columns
    final_data = pd.concat([input_data_scaled, input_data_encoded], axis=1)


    # Replace NaN values with 0
    final_data = final_data.fillna(0)

    #The predicition
    predictions=model.predict(final_data)
    
    print(predictions)
    
    return predictions

    
    
    

@app.route('/predict', methods=['POST'])
#Will return predictions to source
def make_prediction():
    
    scaler = joblib.load('houses_stdscaler.joblib')
    model=joblib.load('houses_gb_model.joblib')
    encoder=joblib.load("houses_encoder.joblib")
    
    # Get the input data from the request
    input_data = request.get_json()

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Call the predict function
    predictions = predict(scaler,encoder ,model, input_df)

    # Return the predictions as a JSON response
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    