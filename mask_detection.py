import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_directory = os.path.join(directory, label)

        if os.path.isdir(label_directory):
            for image_file in os.listdir(label_directory):
                image_path = os.path.join(label_directory, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Load and preprocess the dataset
dataset_path = "path_to_dataset_directory"
images, labels = load_images_from_directory(dataset_path)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=123)

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to numerical values
label_to_index = {"with_mask": 1, "without_mask": 0}
train_labels = np.array([label_to_index[label] for label in train_labels])
test_labels = np.array([label_to_index[label] for label in test_labels])

# Apply data augmentation to the training set
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(1000).batch(32).map(lambda x, y: (data_augmentation(x, training=True), y))

# Build the model
model = keras.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile and train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=10)

# Save the trained model
model.save("mask_detection_model")

# Real-time Mask Detection

# Load the trained model
model = keras.models.load_model("mask_detection_model")

# Define the labels for mask and no mask
LABELS = ["Without Mask", "With Mask"]

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (128, 128))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = face_image / 255.0

        # Predict mask or no mask
        prediction = model.predict(face_image)[0][0]
        label = LABELS[int(prediction > 0.5)]

        # Draw bounding box and label on the frame
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow("Mask Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
