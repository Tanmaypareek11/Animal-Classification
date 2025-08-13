from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)
# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), "animal_classifier.h5")
model = tf.keras.models.load_model(model_path)

# Categories from your dataset
categories = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger'
]


# Prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    index = int(np.argmax(predictions))
    accuracy = round(float(predictions[index] * 100), 2)
    return categories[index], accuracy


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    prediction, accuracy = predict_image(filepath)
    return jsonify({
        "prediction": prediction,
        "accuracy": accuracy,
        "image_url": filepath
    })


if __name__ == "__main__":
    app.run(debug=True)
