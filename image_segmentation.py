import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Step 1: Load the trained model

model_path = 'model.h5'
model = load_model(model_path)
label_to_index = {}  # Dictionary to map label names to numerical indices


# Step 2: Load and preprocess the dataset

def load_dataset(label_to_index):
    images = []
    labels = []

    # Specify the directory containing the images
    directory = 'Kerasimages/Validate'

    # Loop through subdirectories
    index = 0

    for label in os.listdir(directory):
        # Full path to the subdirectory by joining the directory with label
        subdirectory = os.path.join(directory, label)

        if os.path.isdir(subdirectory):
            # Get all image files in the subdirectory
            image_files = os.listdir(subdirectory)

            # Process each image file
            for image_file in image_files:
                # Create the full path to the image file
                image_path = os.path.join(subdirectory, image_file)

                # Load image
                image = cv2.imread(image_path)
                # If the image is None (failed to load), continue to the next image file.
                if image is None:
                    continue

                processed_image = preprocess_image(image)  # preprocess image

                # Append processed image and label to the lists
                images.append(processed_image)

                # Check if the label is already present in the label_to_index dictionary.
                # If not, assign a new index to it and add the mapping to the dictionary.
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1
                labels.append(label_to_index[label])

    # Convert lists to numpy arrays to be used with tensorflow
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.)
    resized_image = cv2.resize(image, (192, 192))  # Resize grid cell to 64x64 pixels
    normalized_image = resized_image / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_image



# Load and preprocess the dataset
images, labels = load_dataset(label_to_index)

# Get the number of unique classes
num_classes = len(np.unique(labels))

# Convert labels to categorical format
labels_categorical = to_categorical(labels, num_classes)

# Step 3: Detect instances of trained images in the larger image

def detect_instances(larger_image_path, threshold):
    larger_image = cv2.imread(larger_image_path)
    processed_larger_image = preprocess_image(larger_image)
    image_height, image_width, _ = processed_larger_image.shape
    predictions = []

    # Calculate the size of each grid cell
    grid_cell_width = image_width // 3
    grid_cell_height = image_height // 3

    for i in range(3):
        for j in range(3):
            # Calculate the coordinates of the grid cell
            x1 = j * grid_cell_width
            y1 = i * grid_cell_height
            x2 = (j + 1) * grid_cell_width
            y2 = (i + 1) * grid_cell_height

            # Extract the grid cell from the larger image
            grid_cell = processed_larger_image[y1:y2, x1:x2]
            processed_grid_cell = np.expand_dims(grid_cell, axis=0)

            # Predict the labels for the grid cell
            predictions_grid_cell = model.predict(processed_grid_cell)
            predicted_class_indices = np.where(predictions_grid_cell[0] > threshold)[0]

            for predicted_class_index in predicted_class_indices:
                predicted_class = [k for k, v in label_to_index.items() if v == predicted_class_index][0]

                # Calculate the coordinates relative to the larger image
                x1_global = x1
                y1_global = y1
                x2_global = x2
                y2_global = y2

                predictions.append((predicted_class, x1_global, y1_global, x2_global, y2_global))

    return predictions



# Step 4: Perform instance detection on the larger image

larger_image_path = 'Images/Image00004.png'
#patch_size = 64  # Adjust this based on the size of your trained images
threshold = 0.5  # Adjust this threshold as needed

detections = detect_instances(larger_image_path, threshold)

# Load the larger image
larger_image = cv2.imread(larger_image_path)


# Step 5: Visualize the detections

for i, (predicted_class, x1, y1, x2, y2) in enumerate(detections):
    cv2.rectangle(larger_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(larger_image, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Detections', larger_image)
cv2.waitKey(0)
cv2.destroyAllWindows()