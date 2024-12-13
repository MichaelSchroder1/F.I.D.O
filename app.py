from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Initialize the Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the EfficientNetB0 model pre-trained on ImageNet
model = EfficientNetB0(weights='imagenet')

@app.route('/')
def home():
    """
    Renders the home page with the upload form.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles file uploads and performs breed prediction using EfficientNetB0.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400  
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make a prediction using EfficientNet
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)  # Get top-1 prediction
        top_prediction = decoded_predictions[0][0]  # Top prediction details
        breed = top_prediction[1]  # Predicted breed 
        confidence = top_prediction[2]  # Confidence score (e.g., 0.95)

        # Check confidence level
        if confidence < 50.0:
            message = (
                "We couldn't confidently identify your doggo's breed. "
                "Try submitting a front-facing, well-lit photo for better accuracy. "
                "But rest assured, your dog is 100% a Good Boy (or Girl)!"
            )
            return jsonify({'breed': breed.replace('_', ' '), 'confidence': float(confidence), 'message': message})
        else:
            return jsonify({'breed': breed.replace('_', ' '), 'confidence': float(confidence)})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'error': "We couldn't recognize your doggo in this photo. Try submitting a front-facing, well-lit photo instead. But don't worry, your dog is 100% a Good Boy (or Girl)!"
        }), 500

if __name__ == '__main__':
    """
    Runs the Flask app in debug mode.
    """
    app.run(debug=True)