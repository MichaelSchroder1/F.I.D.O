###############################################
READ ME  - FIDO
Authur: Michael Schroeder
###############################################


###############################################
GITHUB URL:   
https://github.com/MichaelSchroder1/F.I.D.O
###############################################


###############################################
********ABOUT*********
FIDO: Fetch Identifiable Dog Origins
FIDO is an AI-powered application designed to recognize dog breeds from uploaded images.
This guide outlines the files and folders required to run the program.
###############################################


Required Files and Folders
1. app.py
Purpose: The main application file for FIDO.
Functionality:
Hosts the Flask web server.
Handles image uploads and processes requests.
Uses the EfficientNetB0 model to predict the dog breed and confidence level.
Provides friendly error messages and feedback.
2. templates/index.html
Purpose: Frontend HTML file for the user interface.
Functionality:
Allows users to upload a photo of their dog.
Displays results, including the predicted breed and confidence level.
3. data/ (Folder)
Purpose: Stores uploaded images temporarily for processing.
Setup:
Automatically created by the app if it doesn’t exist.
Ensure the program has write permissions for this directory.
4. Pre-Trained Model
EfficientNetB0:
Pre-trained on ImageNet; no additional model files are required.
Imported directly via TensorFlow/Keras libraries:
python
Copy code
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
5. Dependencies
Ensure the following Python libraries are installed:

Flask
TensorFlow
numpy
To install all dependencies, run:

bash
Copy code
pip install flask tensorflow numpy
Optional Components
Static Folder (static/):

If you want to save and serve processed images, create a static/ folder.
Legacy Files:


Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/FIDO.git
cd FIDO
Start the Flask app:
bash
Copy code
python app.py
Open your browser and go to:
arduino
Copy code
http://127.0.0.1:5000
Upload a photo of your dog and see the breed prediction and confidence level!

Future Enhancements
Training a custom model using the Stanford Dogs Dataset.
Adding more specialized dog breed detection capabilities.
