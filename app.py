from flask import Flask, render_template, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Initialize Flask App
app = Flask(__name__)

# Load Machine Learning Models
models = {
    "breast_cancer": joblib.load("breast_cancer.pkl"),
    "diabetes": joblib.load("diabetes.pkl"),
    "heart": joblib.load("heart.pkl"),
    "kidney": joblib.load("kidney.pkl"),
    "liver": joblib.load("liver.pkl"),
}

# Load Deep Learning Models
models["malaria"] = tf.keras.models.load_model("malaria.h5")
models["pneumonia"] = tf.keras.models.load_model("pneumonia.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        disease = data.get("disease")
        input_features = np.array(data.get("features")).reshape(1, -1)

        if disease not in models:
            return jsonify({"error": "Invalid disease type"}), 400

        model = models[disease]

        if disease in ["malaria", "pneumonia"]:
            input_features = np.expand_dims(input_features, axis=-1)

        prediction = model.predict(input_features)
        result = int(prediction[0] > 0.5) if disease in ["malaria", "pneumonia"] else int(prediction[0])

        return jsonify({"disease": disease, "prediction": result})
    except:
        return jsonify({"error": "Invalid input data"}), 400

@app.route("/malariapredict", methods=['POST'])
def malariapredict():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image'])
            img = img.resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)
            model = models["malaria"]
            pred = np.argmax(model.predict(img)[0])
            return jsonify({"prediction": pred})
    except:
        return jsonify({"error": "Image processing failed"}), 400

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredict():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image']).convert('L')
            img = img.resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0
            model = models["pneumonia"]
            pred = np.argmax(model.predict(img)[0])
            return jsonify({"prediction": pred})
    except:
        return jsonify({"error": "Image processing failed"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
