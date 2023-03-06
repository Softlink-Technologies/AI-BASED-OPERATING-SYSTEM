from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the user
    data = request.get_json(force=True)

    # Make a prediction using the machine learning model
    prediction = model.predict([data['input']])[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
