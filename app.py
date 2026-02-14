import gradio as gr
import numpy as np
import logging
import os
from tensorflow.keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


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


port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)
