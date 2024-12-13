import tensorflow as tf
import numpy as np

# Load the MobileNetV2 model pre-trained on ImageNet
def load_model():
    """
    Loads the pre-trained MobileNetV2 model and customizes it for dog breed detection.

    Returns:
        model: A TensorFlow/Keras model object ready for predictions.
    """
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Add a custom classification head for dog breeds
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')  # Replace 10 with the number of dog breeds
    ])
    
    # Save the model for future use (optional)
    model.save('models/dog_breed_model.h5')
    print("MobileNetV2 model customized and saved!")
    return model


def predict_breed(image, model):
    """
    Predicts the breed of the dog in the given image using the provided model.

    Args:
        image (numpy.ndarray): Preprocessed image as a NumPy array.
        model: Loaded TensorFlow/Keras model.

    Returns:
        dict: Dictionary containing the predicted breed and confidence score.
    """
    try:
        # Add batch dimension to the image (expected input shape for models)
        image = np.expand_dims(image, axis=0)

        # Get predictions
        predictions = model.predict(image)
        confidence = np.max(predictions)  # Highest probability
        breed_index = np.argmax(predictions)  # Index of the highest probability
        breed = get_breed_name(breed_index)  # Map index to breed name

        return {"breed": breed, "confidence": confidence}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def get_breed_name(index):
    """
    Maps an index to a breed name.

    Args:
        index (int): Index of the predicted breed.

    Returns:
        str: Breed name.
    """
    # Example mapping (replace with your actual breed names)
    breed_names = ['Labrador Retriever', 'Poodle', 'German Shepherd', 'Bulldog', 'Beagle', 
                   'Golden Retriever', 'Shih Tzu', 'Chihuahua', 'Pomeranian', 'Dachshund']
    return breed_names[index] if index < len(breed_names) else "Unknown Breed but 100 Percent Good Boy!"