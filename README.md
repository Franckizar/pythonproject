# Traffic Sign Recognition Project

## Overview
This project utilizes a neural network to classify traffic signs. The model is trained on a traffic sign dataset and is capable of predicting the type of traffic sign from a given image. A simple Tkinter-based GUI allows users to upload an image and view the prediction.

## Folder Structure
- **traffic.py**: Python script for training the model.
- **best_model.h5**: The saved version of the best-trained model.
- **predict_sign.py**: Python script with a Tkinter GUI for uploading an image and displaying the prediction.
- **images/**: Directory containing 10 sample traffic sign images (used for testing).
- **other_models/** (optional): Directory for any additional models you may have trained.

## Installation and Setup

1. **Install dependencies**:
    - Install TensorFlow:
      ```bash
      pip install tensorflow
      ```
    - Install Tkinter (if not already installed):
      ```bash
      sudo apt-get install python3-tk
      ```

2. **Clone the repository**:
    - Clone the repository to your local machine:
      ```bash
      git clone https://github.com/Franckizar/pythonproject.git
      ```

## Training the Model

1. The **traffic.py** file contains the code for training the neural network on the traffic sign dataset.
2. Run the following command to train the model and save the trained model as `annotherone_model.h5`:
   ```bash
   python traffic.py
