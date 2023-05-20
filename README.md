# Image Classification with CNN

This code demonstrates a simple image classification task using a Convolutional Neural Network (CNN) implemented in Keras with TensorFlow backend. The purpose of the code is to train a model on a dataset of labeled images and then use the trained model to predict the labels of new images.

## Setup

To set up and run the code, follow these steps:

Ensure that you have the necessary dependencies installed, including TensorFlow, Keras, OpenCV (cv2), and NumPy. If any of these libraries are missing, you can install them using pip install.

Prepare the dataset: The dataset should be organized in subdirectories, where each subdirectory represents a different class. The images belonging to each class should be placed inside their respective subdirectories. Adjust the directory variable in the code to specify the directory path containing the training images.

Data preprocessing: The preprocess_image function resizes each image to a fixed size (64x64 pixels) and normalizes the pixel values to the range [0, 1]. Adjust the image_height, image_width, and num_channels variables according to the desired image dimensions and number of color channels.

Model training: If a saved model file named model.h5 exists, the code loads the model from disk; otherwise, it trains a new model. To change the number of training epochs and batch size, modify the num_epochs and batch_size variables, respectively.

Model evaluation: After training or loading the model, it evaluates the model's performance on the test set.

Predict on new images: Specify the paths of the new images in the new_image_paths list. The code will load and preprocess these images, predict their labels using the trained model, and display the predicted labels.

## Example Usage

Here's an example of how you can use this code:

Organize your image dataset in subdirectories, where each subdirectory represents a different class (e.g., 'A', 'B', 'Circle', etc.).

Adjust the directory path in the code to point to the directory containing your training images.

Run the code. It will train a CNN model on the provided dataset, save the trained model to disk, evaluate the model's performance, and predict the labels of the specified new images.

Check the console output for the predicted labels of the new images.


## Notes

The code assumes that the dataset is split into training and testing sets automatically using an 80-20 ratio. If you prefer a different split, you can modify the test_size parameter in the train_test_split function call.

The code uses categorical labels, meaning that it expects the class labels to be represented as integers. It automatically converts the labels to categorical format using the to_categorical function.

If you rerun the code after training and saving the model, it will load the saved model instead of training a new one. If you want to train a fresh model, delete the model.h5 file before running the code again.

That's it! You should now have a basic understanding of how to set up and use this image classification code. Feel free to modify and expand it to suit your specific needs.