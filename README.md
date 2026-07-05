MNIST Digit Classification using Convolutional Neural Networks (CNN)

This notebook demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

Project Overview
The goal of this project is to accurately classify handwritten digits (0-9) using a deep learning model. The MNIST dataset is a classic benchmark in computer vision, often used as a 'hello world' for image classification.

Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image of a handwritten digit. The dataset is readily available through TensorFlow/Keras.

Steps Performed:
Data Loading & Initial Setup:

Loaded the MNIST dataset using tf.keras.datasets.mnist.load_data().
Unpacked the dataset into training (X_train, y_train) and testing (X_test, y_test) sets.
Inspected data shapes and types.
Data Preprocessing:

Normalization: Image pixel values were normalized from the range [0, 255] to [0.0, 1.0].
One-Hot Encoding: Labels were converted from integer format to one-hot encoded vectors (e.g., 5 becomes [0,0,0,0,0,1,0,0,0,0]).
Visualization: Displayed a sample of preprocessed images and their corresponding labels.
Model Definition & Training:

CNN Architecture: A Sequential CNN model was defined with:
Two Conv2D layers (32 and 64 filters respectively) with relu activation.
Two MaxPooling2D layers to reduce dimensionality.
A Flatten layer to convert 2D feature maps into a 1D vector.
Two Dense layers, including a final output layer with 10 units (for 10 classes) and softmax activation.
Compilation: The model was compiled using the adam optimizer, categorical_crossentropy loss function, and accuracy as the metric.
Training: The model was trained for 10 epochs with a batch size of 32, using the test set for validation.
Model Evaluation:

Visualized training and validation accuracy and loss over epochs.
Evaluated the model's performance on the unseen test set.
Model Saving:

The trained model_cnn was saved in the HDF5 (.h5) format as mnist_model.h5 for future use.
Results
Test Loss: 0.0458
Test Accuracy: 0.9896 (approximately 98.96% accurate)
The model achieved excellent performance on the MNIST digit classification task, demonstrating its ability to learn and generalize from the training data.
