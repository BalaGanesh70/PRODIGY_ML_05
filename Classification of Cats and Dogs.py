import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics
from skimage import io, color, transform

# Define paths to your dataset
cat_path = "path/to/cats"
dog_path = "path/to/dogs"

# Function to load and preprocess images
def load_and_preprocess_images(path, label):
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(path, filename)
            img = io.imread(image_path)
            img = color.rgb2gray(img)  # Convert to grayscale
            img = transform.resize(img, (64, 64))  # Resize image to (64, 64)
            images.append(img.flatten())  # Flatten the image
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess cat images
cat_images, cat_labels = load_and_preprocess_images(cat_path, 0)

# Load and preprocess dog images
dog_images, dog_labels = load_and_preprocess_images(dog_path, 1)

# Concatenate cat and dog data
X = np.concatenate((cat_images, dog_images), axis=0)
y = np.concatenate((cat_labels, dog_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# You can now use svm_model.predict() for classifying new images
# Example: prediction = svm_model.predict(new_image.flatten().reshape(1, -1))
