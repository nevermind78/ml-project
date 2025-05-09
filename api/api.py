import requests
import joblib
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load models once at startup
def load_models():
    # Download models from GitHub (only once)
    def download_model(model_url, model_name):
        response = requests.get(model_url)
        with open(model_name, 'wb') as f:
            f.write(response.content)

    # Download and load models
    download_model('https://github.com/nevermind78/ml-project/raw/refs/heads/main/models/logistic_regression.pkl', 'logistic_regression.pkl')
    download_model('https://github.com/nevermind78/ml-project/raw/refs/heads/main/models/linear_svc.pkl', 'linear_svc.pkl')
    download_model('https://github.com/nevermind78/ml-project/raw/refs/heads/main/models/knn.pkl', 'knn.pkl')

    # Load models into memory
    logistic_regression = joblib.load('logistic_regression.pkl')
    linear_svc = joblib.load('linear_svc.pkl')
    knn = joblib.load('knn.pkl')

    return logistic_regression, linear_svc, knn

# Load models when the API starts
logistic_regression, linear_svc, knn = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file.seek(0)
    
    try:
        file_data = file.read()
        image = Image.open(io.BytesIO(file_data)).convert('L').resize((28, 28))
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    image_array = np.array(image).reshape(1, -1) / 255.0
    model_name = request.form.get('model')

    if model_name == "Logistic Regression" and logistic_regression is not None:
        prediction = logistic_regression.predict(image_array)
    elif model_name == "Linear SVC" and linear_svc is not None:
        prediction = linear_svc.predict(image_array)
    elif model_name == "KNN" and knn is not None:
        prediction = knn.predict(image_array)
    else:
        return jsonify({"error": f"Model '{model_name}' is either invalid or not loaded."}), 400

    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)