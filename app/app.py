from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire detection model', 'fire_detection_model.keras')
model = load_model(os.path.abspath(MODEL_PATH))

# Load class indices (ensure this file is created during training)
with open(os.path.join(os.path.dirname(__file__), '..', 'fire detection model', 'class_indices.json')) as f:
    class_indices = json.load(f)

# Invert class_indices to get class names in index order
class_names = [label for label, idx in sorted(class_indices.items(), key=lambda x: x[1])]

# Upload folder
APP_ROOT = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100  # convert to percentage

        return render_template(
            'index.html',
            prediction=predicted_label,
            confidence=f"{confidence:.2f}%",
            filename=filename
        )
    else:
        return render_template('index.html', prediction="Invalid file type. Allowed: png, jpg, jpeg")

    
# ...existing imports...
from flask import Response
from real_time import gen_frames

# ...your existing code...

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(model, class_names), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

