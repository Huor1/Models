import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)



def predict(normalizer,scaler,label_encoder,encoder,model,input_data):
    
    new_data_normalized = normalizer.fit_transform(input_data[['mileage', 'vol_engine']]) # Apply to numerical columns
    input_data[['mileage', 'vol_engine']] = new_data_normalized

    new_data_scaled = scaler.fit_transform(input_data[['mileage', 'vol_engine']]) # Apply to numerical columns
    input_data[['mileage', 'vol_engine']] = new_data_scaled

    #Preprocess
    high_cardinality_features = ['mark', 'model']
    for feature in high_cardinality_features:
        input_data[feature] = label_encoder.fit_transform(input_data[feature])


    # One-Hot Encoding for low cardinality features
    df_encoded = pd.get_dummies(input_data, columns=['fuel'])

    # Reorder the columns based on the saved encoder
    df_encoded = df_encoded.reindex(columns=encoder)

    df_encoded=df_encoded.drop('price',axis=1)
    # Print the encoded DataFrame
    print(df_encoded)
    
    # Replace NaN values with 0
    df_encoded = df_encoded.fillna(0)

    #The predicition
    predictions=model.predict(df_encoded)
    print(predictions)
    
    
    #    #Using inference on given data
    # input_data = pd.DataFrame({
    #     'mark': ['chevrolet'],
    #     'model': ['cruze'],
        
    #     'year': [2010],
    #     'mileage': [210000],
    #     'vol_engine': [1991],
    #     'fuel': ['Diesel']
    # })
    

@app.route('/predict', methods=['POST'])
def make_prediction():
    # Load the models and encoders
    normalizer = joblib.load('cars_minmaxscaler.joblib')
    scaler = joblib.load('cars_stdscaler.joblib')
    label_encoder = joblib.load('cars_label_encoder.joblib')
    encoder = joblib.load('one_hot_encoder.joblib')
    model = joblib.load('cars_gb_model.joblib')

    # Get the input data from the request
    input_data = request.get_json()

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Call the predict function
    predictions = predict(normalizer, scaler, label_encoder, encoder, model, input_df)

    # Return the predictions as a JSON response
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)