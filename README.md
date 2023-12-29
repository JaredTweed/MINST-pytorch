
# MNIST Digit Recognition and Drawing Application

This project consists of two main components: a Python script for training a Convolutional Neural Network (CNN) on the MNIST dataset and a drawing application built with Pygame that uses the trained model to predict handwritten digits.

## Project Structure

- `train_model.py`: Contains the code for defining, training, and saving the CNN model.
- `drawing_app.py`: A Pygame-based application for drawing digits and predicting them using the trained model.
- `mnist_net.pth`: The saved model state after training.

## train_model.py

This script uses PyTorch to define and train a CNN on the MNIST dataset. The dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). The network includes convolutional layers, batch normalization, dropout layers, and fully connected layers. The model is trained with data augmentation techniques to improve robustness. After training, the model's state is saved to `mnist_net.pth`.

### Model Architecture

The CNN architecture consists of:
- Two convolutional blocks, each with convolutional layers, batch normalization, and dropout.
- Fully connected layers at the end.

### Training Process

- The MNIST dataset is loaded with data augmentation (random affine transformations).
- The network is trained for 15 epochs using the Adam optimizer and a learning rate scheduler.
- The training progress is printed in the console.
- After training, the model's accuracy is tested on the test dataset, and results are printed.
- The trained model state is saved for later use in the drawing application.

### Running the Script

Run the script using:

```bash
python train_model.py
```

This will start the training process, and the trained model will be saved as `mnist_net.pth`.

## drawing_app.py

This Pygame application allows the user to draw a digit on a canvas, and then it predicts the digit using the trained model.

### Features

- A canvas where the user can draw digits using the mouse.
- Real-time digit prediction displayed in the window's title.
- Ability to clear the canvas by pressing the 'c' key.

### Predicting Digits

- The application captures the canvas contents and preprocesses it to match the MNIST dataset format.
- The preprocessed image is fed to the trained model, and the predicted digit is displayed.

### Dependencies

Ensure you have Pygame and PyTorch installed:

```bash
pip install pygame torch torchvision
```

### Running the Application

Run the script using:

```bash
python drawing_app.py
```

Draw a digit on the canvas, and the prediction will be displayed. Press 'c' to clear the canvas and draw a new digit.

## GitHub Repository Contents

- `train_model.py`: The training script for the CNN model.
- `drawing_app.py`: The Pygame application for drawing and digit prediction.
- `mnist_net.pth`: The pre-trained CNN model.

## Conclusion

This project demonstrates the integration of a machine learning model with a Pygame application, providing a simple and interactive way to test the capabilities of a CNN trained on the MNIST dataset.
