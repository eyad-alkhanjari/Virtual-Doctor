import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import time
from sklearn.model_selection import train_test_split

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define relative paths
INPUT_FOLDER = os.path.join(BASE_DIR, 'brain_tumor_dataset', 'INPUT')
MODEL_PATH = os.path.join(BASE_DIR, 'NewForTestingBrainTumor.h5')
path_Yes = os.path.join(BASE_DIR, 'brain_tumor_dataset', 'Data', 'Yes', '*.jpg')
path_No = os.path.join(BASE_DIR, 'brain_tumor_dataset', 'Data', 'No', '*.jpg')
RESULTS_FILE = os.path.join(BASE_DIR, 'results.txt')

# Class names (labels)
class_names = ['no-tumor', 'tumor']

# Function to load and preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to read image {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure color channels match training data
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Load and preprocess images
def load_images(path, label):
    images = []
    labels = []
    for file in glob.iglob(path):
        img = preprocess_image(file)
        if img is not None:
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess all images
images_yes, labels_yes = load_images(path_Yes, 1)
images_no, labels_no = load_images(path_No, 0)

# Check if images are loaded correctly
print(f"Loaded {len(images_yes)} tumor images.")
print(f"Loaded {len(images_no)} no-tumor images.")

# Combine and split data
X = np.concatenate((images_yes, images_no), axis=0)
y = np.concatenate((labels_yes, labels_no), axis=0)

# Ensure there's data to split
if len(X) == 0 or len(y) == 0:
    raise ValueError("No images found. Check the paths and ensure images are available in the directories.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

# Calculate class weights
unique_classes = np.unique(y_train)
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {cls: total_samples / (len(unique_classes) * count) for cls, count in enumerate(class_counts)}

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), class_weight=class_weights)

# Save the model
model.save(MODEL_PATH)

# Load the trained model for prediction
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to predict the class of an image
def predict_image_class(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = int(prediction > 0.5)  # Threshold for binary classification
    return predicted_class, prediction

# Monitor the INPUT_FOLDER for new images and process them
print("Waiting for images...")

while True:
    input_images = os.listdir(INPUT_FOLDER)
    if input_images:
        with open(RESULTS_FILE, 'w') as results_file:
            for image_name in input_images:
                image_path = os.path.join(INPUT_FOLDER, image_name)
                if os.path.isfile(image_path):  # Check if it is a file
                    predicted_class, confidence = predict_image_class(image_path)
                    class_name = class_names[predicted_class]
                    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
                    result_message = f'this is a brain mri scan report tell The patient that you {"have a brain tumor" if class_name == "tumor" else "are fine"} and give recommendations'
                    print(result_message)
                    results_file.write(f'{image_name}: {result_message}\n')
                    # Delete the image after processing
                    os.remove(image_path)
                    results_file.flush()
                    os.fsync(results_file.fileno())
    time.sleep(1)  # Check for new images every second
