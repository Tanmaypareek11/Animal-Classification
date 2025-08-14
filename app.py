import gradio as gr
import pickle
from PIL import Image
import numpy as np

# Load the model
with open("animal_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict(image):
    # Resize image (adjust size according to your model)
    image = image.resize((64, 64))
    # Convert to array and flatten if needed
    image_array = np.array(image).reshape(1, -1)
    # Make prediction
    pred = model.predict(image_array)[0]
    return pred

# Build Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label()
)

# Launch (locally for testing)
if __name__ == "__main__":
    iface.launch()
