# Flask app.py

from flask import Flask, request, render_template
import numpy as np
import pickle
import io
from PIL import Image
import base64

app = Flask(__name__)
app.jinja_env.cache = {}

# Load the pickled model
with open("best.pkl", "rb") as f:
    model = pickle.load(f)

# Define class names
class_names = ["Anthracnose", "algal leaf", "bird eye spot", "brown blight", "gray light", "healthy", "red leaf spot", "white spot"]

# Define image dimensions
img_height = 224
img_width = 224

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files["image"]

        # Read the image as bytes
        image_bytes = image_file.read()

        # Convert the image to a NumPy array
        external_image = Image.open(io.BytesIO(image_bytes)).resize((img_height, img_width))
        external_image_array = np.array(external_image)

        # Make prediction without normalization or preprocessing
        predictions = model.predict(np.expand_dims(external_image_array, axis=0))

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        # Convert the image to Base64
        image_data_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Render a new template for displaying the result
        return render_template("result.html", predicted_disease=predicted_class_name, image_data_base64=image_data_base64)

    except Exception as e:
        # Handle errors
        print(f"Error: {e}")
        return render_template("result.html", error=str(e))

if __name__ == '__main__':
    app.run(debug=True)