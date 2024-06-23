from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from JSON request
    data = request.get_json()
    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    age = int(data['age'])
    
    # Make prediction using the loaded model
    prediction = model.predict([[area, bedrooms, age]])
    
    # Return prediction as JSON response
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
