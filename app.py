import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL importimport gradio as gr
import numpy as np
import logging
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------
# Logging Setup
# ------------------------
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ------------------------
# Load Model
# ------------------------
model = load_model("animal_classifier.h5")

categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer',
              'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse']


# ------------------------
# Prediction Function
# ------------------------
def predict_animal(img):
    try:
        img_display = img.copy()

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100

        # Top 3 predictions
        top_3 = prediction.argsort()[-3:][::-1]
        probs_dict = {categories[i]: float(prediction[i]) for i in top_3}

        # Confidence threshold
        if confidence < 50:
            result_text = "Model is uncertain. Try another image."
        else:
            result_text = f"Predicted: {categories[class_idx]} ({confidence:.2f}%)"

        logging.info(f"Prediction: {categories[class_idx]} - {confidence:.2f}%")

        return img_display, result_text, probs_dict

    except Exception as e:
        return None, f"Error: {str(e)}", {}


# ------------------------
# Gradio UI (Professional)
# ------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐾 Animal Classification AI")
    gr.Markdown("Upload an animal image and get top 3 predictions with confidence.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        image_output = gr.Image(label="Preview")

    prediction_text = gr.Textbox(label="Prediction")
    label_output = gr.Label(label="Top Predictions")

    btn = gr.Button("Predict")

    btn.click(
        predict_animal,
        inputs=image_input,
        outputs=[image_output, prediction_text, label_output]
    )

demo.launch()
 Image

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
