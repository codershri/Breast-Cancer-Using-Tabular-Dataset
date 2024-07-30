from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize and fit the scaler
scaler = StandardScaler()

# Load the training data to fit the scaler (if available)
# Replace 'X_train.npy' with the filename of your training data
# Assuming your training data is in the form of a numpy array
X_train = np.load('X_train.npy')  # Adjust the filename accordingly
scaler.fit(X_train)

def make_prediction(input_data):
    # Reshape and standardize the input data
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = model.predict(input_data_std)

    # Convert prediction to text
    if prediction[0] == 0:
        result = 'Malignant'
    else:
        result = 'Benign'

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = [float(x) for x in request.form.values()]

    # Call the make_prediction function
    prediction_text = make_prediction(input_data)

    return render_template('index.html', prediction_text='The tumor is {}'.format(prediction_text))

if __name__ == '__main__':
    app.run(debug=True)
