import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

IMG_WIDTH = 30
IMG_HEIGHT = 30

# Load the trained model
model_path = "C:/Users/TAKAM/Downloads/AIProject-master/AIProject-master/BalembaJessy/traffic_model.h5"
model = tf.keras.models.load_model(model_path)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_class, confidence = predict_image(file_path)

        # Print to console for debugging
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")  
        
        result_label.config(text=f"Predicted: {predicted_class}, Confidence: {confidence:.2f}%")

        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize for display
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Create GUI
root = tk.Tk()
root.title("Traffic Sign Classifier")

# Set the background color of the window
root.configure(bg="#f0f0f0")

# Frame for the button and labels with padding
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=30)

# Button with a more stylish design
btn = tk.Button(frame, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#4CAF50", fg="white", relief="flat", width=20, height=2)
btn.pack()

# Label for displaying the image
img_label = tk.Label(root, bg="#f0f0f0")
img_label.pack(pady=20)

# Result label with a more prominent style
result_label = tk.Label(root, text="Upload an image to classify", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333")
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
