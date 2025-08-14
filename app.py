import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model("animal_classifier.h5")

# Class names
categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
              'Elephant', 'Giraffe', 'Horse']


# Prediction function
def predict_animal(img):
    img_display = img.copy()

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]  # probabilities
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx] * 100

    # Return dictionary of probabilities for Gradio Label output
    probs_dict = {categories[i]: float(prediction[i]) for i in range(len(categories))}

    return img_display, f"Predicted: {categories[class_idx]} ({confidence:.2f}%)", probs_dict


# Gradio interface
iface = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(), gr.Textbox(), gr.Label(num_top_classes=len(categories))],
    title="Animal Classification",
    description="Upload an image of an animal and the model will predict its class with confidence percentages."
)

iface.launch()
