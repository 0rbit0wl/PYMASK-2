# Mask Detection Script

This script uses a trained mask detection model to perform real-time mask detection using your computer's camera. It utilizes the OpenCV library for video capture and face detection, and TensorFlow-Keras for model loading and inference.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV

## Usage

1. Install the required packages using the following command:
           pip install -r requirements.txt or python3 -m pip install -r requirements.txt

2. Ensure that you have a trained mask detection model saved in a file named `mask_detection_model`. If you don't have a trained model, follow the instructions in the main `README.md` file to train a model and save it.

3. Run the `mask_detection.py` script to start real-time mask detection using your computer's camera:


A new window will open displaying the camera feed, and the script will draw bounding boxes and labels on faces indicating whether a person is wearing a mask or not. Press 'q' to exit the program.

4. Customize the script as needed:
- Adjust the `LABELS` list to match the labels used in your trained model.
- Modify the face detection parameters in the script to suit your specific use case, such as the scale factor, minimum neighbors, and minimum size.

## Acknowledgments

- The mask detection script is built using concepts and techniques from the fields of computer vision and deep learning.
- The performance of the mask detection relies on the quality and diversity of the trained model and the camera feed.
- This script serves as a demonstration and starting point for real-time mask detection applications.

## License

This script is licensed under the [MIT License](LICENSE).
